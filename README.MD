# Ascend 200I DK A2的部署仓库
---
## 配置网络(采用Type-C连接)
### 1. 烧录网口配置Type-C的ip地址为：*192.168.0.2*，子网掩码：*255.255.225.0*，ssh连接时采用烧录时的*192.168.0.2*
### 2. 通过笔记本的C口跟开发板连接，同时配置对应的IP地址以及子网掩码；
## 准备工作
### 1. ATC指令转换格式，将ckpt文件转成om格式进行推理；(转换之前先挂载swap分区避免卡死);
### 2. 本次部署采用的指令为：

```
atc --model=fourclasses.onnx --framework=5 --output=fourclasses --input_format=NCHW --input_shape="x:1,12,320,320" --log=error --soc_version=Ascend310B1 --output_type=FP16 --input_fp16_nodes='x'
```

其中转换参数的含义为：  
* --model：输入模型路径
* --framework：原始网络模型框架类型，5表示ONNX
* --output：输出模型路径
* --input_format：输入Tensor的内存排列方式
* --input_shape：指定模型输入数据的shape
* --log：日志级别
* --soc_version：昇腾AI处理器型号
* --input_fp16_nodes：指定输入数据类型为FP16的输入节点名称
* --output_type：指定网络输出数据类型或指定某个输出节点的输出类型

### 3. 得到om模型后修改对应的部署代码开始执行推理；

## Python基础实现(main_python.ipynb)
### Step1：模型图片预处理，这边要把输入图片的类型转成与模型相对应的(1,12,320,320)，因为yolov5s中原始640*640*3的图像输入Focus结构，采用切片操作，先变成320*320*12的特征图，转成tensor类型处理时要转成float16对应；
### Step2: 载入对应的om文件以及输入img，得到output，这边output包括了大中小三个锚框，其中19200 = anchor_num(3) * grid_h(80) * grid_w(80),大框的尺寸是80*80；然后按照输入图片的尺寸做相应的转换，这边orig_h = orig_w = 640,所以输入图片的尺寸要做调整对应640*640，否则出现框对不上！
### Step3: 模型输出后处理(非极大值抑制后处理以及画框显示), 
```
cfg = {
    'conf_thres': 0.25,  # 模型置信度阈值，阈值越低，得到的预测框越多
    'iou_thres': 0.4,  # IOU阈值，高于这个阈值的重叠预测框会被过滤掉
    'input_shape': [640,640],  # 模型输入尺寸
}
```
## C++基础实现(main_python.ipynb)
### 1. 首先是aipp的配置使用，可以提前对AI推理的输入数据进行预处理的模块;所以这边要对我们的模型的ckpt转成onnx,ckpt2onnx.py代码，转成类型应该是 1，12，320，320
---
---
---
## Modelarts训练模型，采用mindspore 2.0.0alpha的镜像，yolov5对我们数据进行分布式训练；
首先创建好8卡的资源后通过notebook连接服务器；
### 1. git载入对应的MindSpore的版本的模型，放在根目录；
``` 
git clone https://gitee.com/mindspore/models.git -b r2.0.0-alpha
cp -r models/official/cv/YOLOv5 ./
```
### 2. 通过modelarts的obs工具上传我们的数据，并且通过python来解压，因为自带的unzip指令无法解压超过5G的数据，并且数据与coco2017格式对应；
```
import zipfile
f = zipfile.ZipFile("/home/ma-user/work/images.zip")# 压缩文件位置
for file in f.namelist():
    f.extract(file,"/home/ma-user/work/data9705-2/")#解压位置
f.close()
```
### 3. 对YOLOv5的源码进行修改，这边需要格外小心：
其中需要修改的地方有：  
* **配置文件:** YOLOX有一个配置的yaml文件，需要在其中指定数据的类别数为5，以及out_channel = 3 * (num_classes + 5)；
* **标签文件:** 修改数据集的标签文件，确保类别与数据集匹配，路径要进行设置，设置为```"/home/ma-user/work/data9705-2/images/"```，路径有下train2017,val2017,annotations三个文件夹；
* **类别名:** labels以及对应的id要换成我们的四类别以及id;
### 4. 启动分布式训练
* 装map加速模块；
```
pip install pybind11
cd third_party&&bash build.sh
```
* 装库；
```
pip install pycocotools==2.0.5
```
* 启动指令，其中多卡hccl通信文件目录在/user/config/nbstart_hccl.json；
```as
cd scripts
bash ./run_distribute_train.sh /home/ma-user/work/data9705-2/images /user/config/nbstart_hccl.json 
```
---
---
## Modelarts训练Mindyolo，采用mindspore 1.9.1的镜像；
首先创建好8卡的资源后通过notebook连接服务器；
### 1. git载入对应的MindSpore的版本的模型，放在根目录；
```
git clone https://github.com/mindspore-lab/mindyolo.git -b r0.1 #MindSpore2.1之前不支持在910B上运行，另外910B的支持在逐步添加中,因此用r0.1的分支对应1.9.1的版本；
```
### 2. 从obs桶加载数据集，并且要对数据进行处理，符合mindyolo的格式；
* 