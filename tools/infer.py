# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import cv2
import numpy as np
import math
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine import Engine

if __name__ == "__main__":
    args = config.parse_args()
    args.config = r"ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_large_x1_25.yaml"
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    engine = Engine(config, mode="infer")

    with_ = "dataset/1126/train/WithMask/35ARXVI2ND.png"
    out = "dataset/1126/train/WithoutMask/8BVWFFLSZS.png"
    test1 = "output/test1.jpg"
    test2 = "output/test2.jpg"
    test3 = "output/test3_star.jpg"
    test4 = "output/test4_star.jpg"
    # with open(test4, "rb") as f:
    #     frame = f.read()

    # engine.infer(frame)

    cap = cv2.VideoCapture(0)

    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    from hub import PyramidBoxLiteMobile
    model = PyramidBoxLiteMobile()

    while(1):
        # get a frame
        ret, frame = cap.read()
        if not ret:
            continue
        # faces = face_cascade.detectMultiScale(frame, 1.3)
        faces = model.face_detection(images=[frame])
        # faces = [(item['top'], item['bottom'], item['left'], item['right']) for item in faces]
        faces = [(item['left'], 
                  item['top'], 
                  item['right'], 
                  item['bottom']) for item in faces[0]["data"]]

        # for x, y, w, h in faces:
        for x1, y1, x2, y2 in faces:
            
            h, w = y2-y1, x2-x1
            x, y = x1, y1

            curr_img = frame[y:y+h, x:x+w]
            # curr_img = frame[y1:y2, x1:x2]

            res = engine.infer(curr_img)

            frame = cv2.putText(frame, res[0]['label_names'][0], (x+2, y+2), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
            frame = cv2.putText(frame, "%.2f%%"%(res[0]['scores'][0]*100), (x+2, y+22), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)             
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # show a frame
        cv2.imshow("mask", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


