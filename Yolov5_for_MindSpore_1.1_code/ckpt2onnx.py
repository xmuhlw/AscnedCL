# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import argparse
import os

import mindspore
import numpy as np
from mindspore import Tensor, context
from mindspore.train.serialization import (export, load_checkpoint,
                                           load_param_into_net)
from src.config import ConfigYOLOV5
from src.yolo import YOLOV5s, YOLOV5s_Infer

parser = argparse.ArgumentParser(description='yolov5 export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--testing_shape", type=int,
                    default=640, help="test shape")
# parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str,
                    default="oursonnx640", help="output file name.")
# parser.add_argument('--file_format', type=str, choices=["AIR", "MINDIR"], default='AIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="CPU",
                    help="device target")
args = parser.parse_args()
#
# context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
# if args.device_target == "Ascend":
#     context.set_context(device_id=args.device_id)
#
# 创建文件夹，将模型保存至此文件夹下

# 加载ckpt模型，注意如果此行报错，可将cfg.directory,后的代码改为已存在的ckpt文件，例如'resnet-ai_2-150_113.ckpt'
param_dict = load_checkpoint(
    os.path.join(r'D:\WeChat Files\huliwei555\FileStorage\File\2023-09\Yolov5_for_MindSpore_1.1_code\Yolov5_for_MindSpore_1.1_code\0-1000_459000.ckpt'))

# 设置训练网络，加载模型参数到训练网络内，这里以Resnet50为例
# 设置ResNet50网络
resnet = YOLOV5s(is_training=False)

# 加载模型参数到ResNet50网络内
load_param_into_net(resnet, param_dict)
config = ConfigYOLOV5()
if args.testing_shape:
    config.test_img_shape = [int(args.testing_shape), int(args.testing_shape)]
ts_shape = config.test_img_shape[0]

network = YOLOV5s_Infer(config.test_img_shape)
load_param_into_net(network, param_dict)

print(int(ts_shape/2))
input_data = Tensor(np.zeros([1, 3, 640, 640]), mindspore.float32)

export(network, input_data, file_name=args.file_name, file_format='ONNX')

# 导出ONNX模型，设置网络，网络的输入，模型名称，保存格式
# export(resnet, Tensor(x), file_name='./flowers/best_model.onnx', file_format='ONNX')
