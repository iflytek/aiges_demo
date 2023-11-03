#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@file: wrapper.py
@time: 2022-08-19 02:05:07.467170
@project: stream_test
@project: ./
"""
import json
import os.path
import queue

from aiges.core.types import *

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls, SessionCreateResponse, callback  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls, SessionCreateResponse, callback

from aiges.sdk import WrapperBase, \
    StringParamField, \
    ImageBodyField, \
    StringBodyField, IntegerParamField

from aiges.stream import StreamHandleThread
from aiges.utils.log import log, getFileLogger

# 导入inference.py中的依赖包
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import time


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = ImageBodyField(key="img", path="test_data/0.png")
    param1 = StringParamField(key="ctl", value="ok")


# 定义模型的输出参数
class UserResponse(object):
    accept1 = StringBodyField(key="result")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# 定义服务推理逻辑
class Wrapper(WrapperBase):
    serviceId = "mnist"
    version = "v1"
    requestCls = UserRequest()
    responseCls = UserResponse()
    model = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = None
        self.device = None
        self.filelogger = None

    def wrapperInit(self, config: {}) -> int:
        log.info("Initializing ...")

        Wrapper.session_total = config.get("common.lic", 1)
        self.session.init_wrapper_config(config)
        self.session.init_handle_pool("thread", 1, MyReqDataThread)

        return 0

    def wrapperWrite(self, handle: str, req: DataListCls, sid: str) -> int:
        """
        会话模式下: 上行数据写入接口
        :param handle: 会话handle 字符串
        :param req:  请求数据结构
        :param sid:  请求会话ID
        :return:
        """
        _session = self.session.get_session(handle=handle)
        if _session == None:
            log.info("can't get this handle:" % handle)
            return -1
        _session.in_q.put(req)
        print("sending")
        return 0

    def wrapperCreate(self, params: {}, sid: str, persId: int = 0) -> SessionCreateResponse:
        """
        非会话模式计算接口,对应oneShot请求,可能存在并发调用
        @param ret wrapperOnceExec返回的response中的error_code 将会被自动传入本函数并通过http响应返回给最终用户
        @return
            SessionCreateResponse类, 如果返回不是该类会报错
        """
        print(1111)
        s = SessionCreateResponse()
        # 这里是取 handle
        handle = self.session.get_idle_handle()
        if not handle:
            s.error_code = -1
            s.handle = ""
            return s

        _session = self.session.get_session(handle=handle)
        if _session == None:
            log.info("can't create this handle:" % handle)
            return -1
        _session.setup_sid(sid)
        _session.setup_params(params)
        _session.setup_callback_fn(callback)

        print(sid)
        s = SessionCreateResponse()
        s.handle = handle
        s.error_code = 0
        return s

    def wrapperLoadRes(self, reqData: DataListCls, resId: int) -> int:
        return 0

    def wrapperUnloadRes(self, resId: int) -> int:
        return 0

    def wrapperOnceExec(self, params: {}, reqData: DataListCls, persId: int = 0) -> Response:
        pass
        return None

    def wrapperFini(cls) -> int:
        return 0

    def wrapperError(cls, ret: int) -> str:
        if ret == 100:
            return "user error defined here"
        return ""

    '''
        此函数保留测试用，不可删除
    '''

    def wrapperTestFunc(cls, data: [], respData: []):
        pass


class MyReqDataThread(StreamHandleThread):
    """
    流式示例 thread，
    """

    def __init__(self, session_thread, in_q, out_q):
        super().__init__(session_thread, in_q, out_q)
        self.setDaemon(True)
        self.is_stopping = False

    def init_model(self, *args, **kwargs):
        log.info("inting model ...")
        device = torch.device('cuda')
        self.device = torch.device("cpu")
        self.filelogger = getFileLogger()
        self.model = Net().to(self.device)
        ptfile = os.path.join(os.path.dirname("."), "train", "mnist_cnn.pt")
        self.model.load_state_dict(torch.load(ptfile))  # 根据模型结构，调用存储的模型参数
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def stop(self):
        self.is_stopping = True

    def run(self):
        self.init_model(self.session_thread.handle)
        while not self.is_stopping:
            try:
                req = self.in_q.get(timeout=5)
                print(self.session_thread.params)
                self.infer(req)
            except queue.Empty as e:
                pass

    def infer(self, req: DataListCls):
        params = self.session_thread.params
        # 读取测试图片并进行模型推理
        self.filelogger.info("got reqdata , %s" % req.list)
        imagebytes = req.get("img").data

        img = Image.open(io.BytesIO(imagebytes))
        first = True
        steps = 20
        for step in range(steps):
            img1 = self.transform(img).unsqueeze(0)
            print(img1.shape)

            img1.to(self.device)
            result = self.model(img1).argmax()
            log.info("##result ###:%d" % int(result))

            retC = {
                "result": int(result),
                "msg": "result is: %d" % int(result)
            }
            # 使用Response封装result
            res = Response()
            resd = ResponseData()
            resd.key = "result"
            resd.key = 'output_img'
            if step == steps - 1:  # 这里待优化
                resd.status = DataEnd  # 最后一条数据是 2
            elif first:
                resd.status = DataBegin  # 首此返回是0
                first = False
            else:
                resd.status = DataContinue  # 中间数据是1
            resd.type = DataText
            resd.setData(json.dumps(retC).encode("utf-8"))
            #resd.len = len(resd.data)
            res.list = [resd]
            self.session_thread.callback_fn(res, self.session_thread.sid)
            if resd.status == DataEnd:  # 这里要在 i % opt.decoder_step == 0: 外部判断 ，否则进不到这个逻辑
                self.filelogger.info("reseting session")
                self.session_thread.reset()


if __name__ == '__main__':
    m = Wrapper()
    m.run_stream()
