import onnx
import torch
import torch.nn

model = torch.load(r'D:\Desktop\华为项目\ascendCL\RefineDet\RefineDet\best.pt')
model.eval()
input_names = ['input']
output_names = ['output']
 
x = torch.randn(1,3,640,640,requires_grad=True)
 
torch.onnx.export(model, x, 'best.onnx', input_names=input_names, output_names=output_names, verbose='True')