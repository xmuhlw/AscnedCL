

import onnx

from onnxsim import simplify

ONNX_MODEL_PATH = 'E:/下载/our_onnx.onnx'
ONNX_SIM_MODEL_PATH = './our_onnx_simple.onnx'

onnx_model = onnx.load(ONNX_MODEL_PATH)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, ONNX_SIM_MODEL_PATH)
print('finished exporting onnx')
