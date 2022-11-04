#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@file: wrapper.py
@time: 2022-08-19 02:05:07.467170
@project: mnist
@project: ./
"""

import sys
import hashlib
from aiges.types import *
try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls

from aiges.sdk import WrapperBase, \
    StringParamField, \
    ImageBodyField, \
    StringBodyField
from aiges.utils.log import log


# 导入inference.py中的依赖包
import cv2
import torch
import torchvision.transforms as transforms
from Model import MNIST
import numpy as np


# 定义模型的超参数和输入参数
class UserRequest(object):

    input1 = ImageBodyField(key="img", path="test_data/0.png")

# 定义模型的输出参数
class UserResponse(object):
    accept1 = StringBodyField(key=b"number")

# 定义服务推理逻辑
class Wrapper(WrapperBase):
    serviceId = "mnist"
    version = "v1"
    requestCls = UserRequest()
    responseCls = UserResponse()
    model = None

    def wrapperInit(cls, config: {}) -> int:
        log.info("Initializing ...")
        device = torch.device('cpu')
        cls.model = MNIST().to(device)
        cls.model.load_state_dict(torch.load('/Users/yangyanbo/projects/iflytek/code/athenaloader/aiges_demo/mnist/wrapper/mnist.pkl'))
        return 0


    def wrapperOnceExec(cls, params: {}, reqData: DataListCls) -> Response:

        # 读取测试图片并进行模型推理
        log.info("got reqdata , %s" % reqData.list)
        imagebytes = reqData.get("img").data
        image  = [cv2.imdecode(np.frombuffer(imagebytes, np.uint8), cv2.COLOR_BGR2GRAY)]
        image_tensor = torch.unsqueeze(torch.Tensor(image), dim=0)
        result = cls.model(image_tensor).argmax()
        print(result)
        
        # 使用Response封装result
        res = Response()
        resd = ResponseData()
        resd.key = "img"
        resd.type = DataText
        resd.status = Once
        resd.data = result.numpy().tobytes()
        resd.len = len(resd.data)
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
