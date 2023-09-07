#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@file: wrapper.py
@time: 2022-11-15 13:10:10.347339
@project: aiges_huggingface
@project: ./
"""

import io
import os
import json
try:
    from aiges_embed import (
        ResponseData,
        Response,
        DataListCls,
    )
except ImportError:
    from aiges.dto import (
        Response,
        ResponseData,
        DataListCls,
    )

from aiges.sdk import (
    WrapperBase,
    ImageBodyField,
    StringBodyField,
)
from aiges.utils.log import log
from aiges.types import *

from PIL import Image
from transformers import pipeline


current_path = os.path.dirname(os.path.abspath(__file__))


class UserRequest(object):
    input_image = ImageBodyField(
        key="image",
        path=os.path.join(current_path, "test_data/test.png"),
    )
    input_question = StringBodyField(
        key="question",
        value=b"how many dogs here?",
    )


class UserResponse(object):
    accept_outputs = StringBodyField(key="outputs")


class Wrapper(WrapperBase):
    serviceId = "visual-question-answering"
    version = "0.1"
    requestCls = UserRequest()
    responseCls = UserResponse()

    def wrapperInit(self, config: {}) -> int:
        log.info(f"Initializing with config {config}")
        self._model = pipeline("visual-question-answering")
        return 0

    def _to_PIL(self, image: ImageBodyField) -> "Image.Image":
        buf = io.BytesIO(image.data)
        return Image.open(buf)

    def wrapperLoadRes(self, reqData: DataListCls, resId: int) -> int:
        return 0

    def wrapperUnloadRes(self, resId: int) -> int:
        return 0
    
    def wrapperOnceExec(self, params: {}, reqData: DataListCls, persId: int = 0) -> Response:
        log.info("got reqdata , %s" % reqData.list)
        image = reqData.get("image")
        question = reqData.get("question")

        resp = Response()
        if not image:
            log.error("image is required")
            return resp.response_err(100)
        if not question:
            log.error("question is required")
            return resp.response_err(100)
        output = self._model(
            self._to_PIL(image),
            question.data.decode(),
        )
        output_str = json.dumps(output)

        output_data = ResponseData()
        output_data.type = DataText
        output_data.key = "outputs"
        output_data.data = output_str.encode()
        output_data.len = len(output_str.encode())
        output_data.status = Once

        resp.list = [output_data]
        return resp

    def wrapperFini(self) -> int:
        return 0


    def wrapperError(self, ret: int) -> str:
        if ret == 100:
            return "user error defined here"
        return ""

    def wrapperTestFunc(self, data: [], respData: []):
        mock_output = '{"text": "1", "score": 0.9}'
        response_data = ResponseData(
            key="outputs",
            status=1,
            len=len(mock_output),
            data=mock_output,
        )
        resp = Response(
            list=[response_data]
        )
        return resp


if __name__ == '__main__':
    m = Wrapper()
    m.schema()
    m.run()