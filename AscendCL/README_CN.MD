 # 模型训练
 ### 训练环境配置--yolov7：
 #### 1) 采用官方库并配置环境，根据requirements.txt装相关的库；
 ```
 git clone https://github.com/WongKinYiu/yolov7.git
 ```
 #### 2) 修改 ./yolov7/data/coco.yaml，将对应的数据集路径，labels，类别数做调整，如果数据集格式COCO可以采用整理好的一些工具转成YOLO格式
 #### 3）模型执行训练，可以加载预训练权重，把pretrain设置为严格加载即可；
 ```
 python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
 ```
 #### 4）模型转成ONNX导出，这边不带NMS后处理；
 ```
 python export.py --weights best.pt --grid --simplify  --topk-all 100 --img-size 640 640 --max-wh 640
 ```

# AscendCL部署
---
* 推理设备:Atlas 200I DK A2，—soc_version=310B4;
* CANN版本：6.2.RC2
* Ubuntu：22.04，ARM64
* 固件与驱动版本：23.0.RC2
---
参考[MindX推理](https://gitee.com/ascend/samples/wikis/%E9%80%9A%E7%94%A8%E7%9B%AE%E6%A0%87%E8%AF%86%E5%88%AB%E6%A0%B7%E4%BE%8B/%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87/%E4%BE%9D%E8%B5%96%E5%AE%89%E8%A3%85)实现在310B4上采用Yolo模型对多路视频进行推理；
---
*ToDo*： 能否通过PresentAgent将推理结果发送至网页端进行展示；
---
## 实现步骤:
### 1. 烧录网口配置默认，采用以太网口连接至开发板，开启网络共享，具体参考 [网络配置文档](https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC1/Hardware%20Interfaces/hiug/hiug_0010.html)，开发板登ip:192.168.137.100,登录账号：root，登录密码：Mind@123

### 开发环境配置：
#### 1）
a.配置环境变量
```
 export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
 export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
 export THIRDPART_PATH=${DDK_PATH}/thirdpart
 export LD_LIBRARY_PATH=${THIRDPART_PATH}/lib:$LD_LIBRARY_PATH
```
b.将仓库common拷贝至thirdpart；作为第三方依赖
```
cd ${HOME}
sudo apt-get install git
git clone https://gitee.com/ascend/samples.git
# 拷贝公共文件到samples相关依赖路径中
cp -r ${HOME}/samples/common ${THIRDPART_PATH}
```
#### 2) 如果apt-get无法连接至服务器，可以采用换源；
```
vi /etc/apt/source.list #可以把文件备份后换一些国内的清华、中科大源等；
sudo apt-get update #更新
```
#### 3) libopencv-dev安装；默认安装位置为/usr/include/opencv4/opencv2,
```
sudo apt-get install libopencv-dev
sudo cp -r /usr/include/opencv4/opencv2 /usr.
```
#### 2) x264编译安装
```
cd ${HOME}
git clone https://code.videolan.org/videolan/x264.git
cd x264
# 安装x264
./configure --enable-shared --disable-asm
make
sudo make install
sudo cp /usr/local/lib/libx264.so.164 /lib
```
#### 3) ffmpeg编译安装
```
cd ${HOME}
wget http://www.ffmpeg.org/releases/ffmpeg-4.1.3.tar.gz --no-check-certificate
tar -zxvf ffmpeg-4.4.4.tar.gz
cd ffmpeg-4.4.4
# 安装ffmpeg
./configure --enable-shared --enable-pic --enable-static --disable-x86asm --enable-libx264 --enable-gpl --prefix=${THIRDPART_PATH}
make -j8
make install
```
#### 4) aclite编译安装
执行以下命令安装acllite（注意，安装前需要先进行ffmpeg的源码安装）。
```
cd ${HOME}/samples/cplusplus/common/acllite/
make
make install
```
### 执行编译：
```
cd scripts
bash sample_build.sh
```
* X11头文件缺失，sudo apt-get install X11编译安装即可；
* 在main.cpp声明#include <fstream>

### 运行程序：
```
bash sample_run.sh
```
* 采用ffmpeg将MP4格式转为h264，并且通过live555推流，相应教程可以参考[configDemo](D:\Desktop\华为项目\AscendCL\AscendCL\sampleYOLOV7MultiInput\configDemo.md)
```
首先将mp4文件放在/live/mediaServer
执行指令转换格式:ffmpeg -i test.mp4 -c h264 test.h264/ffmpeg -i test.mp4 -c h264 test.264
执行 ./live555MediaServer
在scripts的test.json文件改对应的input_data以及input_type即可
```
### 替换自己的模型：
#### 1）得到训练的ONNX后执行atc转换指令，挂载aipp做处理，--soc_version=Ascend310B4，--framework=5;
#### 2）推理代码修改
* /inc/lables.h，标签头文件修改为自己的类别，NC设置为5；
* /src/detectPostprocess.cpp，classNum做修改，对图像后处理部分进行优化调整;



