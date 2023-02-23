#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@file: wrapper.py
@time: 2022-08-19 02:05:07.467170
@project: mnist
@project: ./
"""
import json
import os.path

from aiges.core.types import *

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls

from aiges.sdk import WrapperBase, \
    ImageBodyField, \
    StringBodyField
from aiges.utils.log import log, getFileLogger

# 导入inference.py中的依赖包
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = ImageBodyField(key="img", path="test_data/0.png")


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
        self.device = torch.device("cpu")
        self.filelogger = getFileLogger()
        self.model = Net().to(self.device)
        ptfile = os.path.join(os.path.dirname(__file__), "train", "mnist_cnn.pt")
        self.model.load_state_dict(torch.load(ptfile))  # 根据模型结构，调用存储的模型参数
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return 0

    def wrapperOnceExec(self, params: {}, reqData: DataListCls) -> Response:
        # 读取测试图片并进行模型推理
        self.filelogger.info("got reqdata , %s" % reqData.list)
        imagebytes = reqData.get("img").data

        img = Image.open(io.BytesIO(imagebytes))

        img = self.transform(img).unsqueeze(0)
        print(img.shape)
        img.to(self.device)
        result = self.model(img).argmax()
        log.info("##result ###:%d" % int(result))

        retC = {
            "result": int(result),
            "msg": "result is: %d" % int(result)
        }
        # 使用Response封装result
        res = Response()
        resd = ResponseData()
        resd.key = "result"
        resd.setDataType(DataText)
        resd.status = Once
        resd.setData(json.dumps(retC).encode("utf-8"))
        res.list = [resd]
        return res

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


if __name__ == '__main__':
    m = Wrapper()
    m.run()
