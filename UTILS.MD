### 1. 导入MindSpore镜像至modelarts；
* 注册一个华为云服务器，这样可以快速拉取/推送镜像；
* 指令安装docker
```
curl -fsSL get.docker.com -o get-docker.sh
sh get-docker.sh
sudo systemctl daemon-reload
sudo systemctl restart docker
```
然后登录昇腾的开源社区```http://mirrors.cn-central-221.ovaijisuan.com/```找到对应的版本
```
基本的docker指令
ocker pull swr.cn-central-221.ovaijisuan.com/wuh-aicc_dxy/chatglm:v7chatglm:v7 #拉取镜像；
docker images #列出镜像；
docker tag #打标签
docker push XXX #推送到对应的计算中心:
docker rmi XXX #删除
```

### 1. 导入MindSpore镜像至modelarts；