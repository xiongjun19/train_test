# train_test 说明

## 环境安装
### 安装docker 和Nvidia docker
如果docker环境存在，则可以跳过下面的内容；
#### 安装docker
```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

sudo groupadd docker
sudo usermod -aG docker $USER
```

#### 安装Nvidia docker
参考 https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide

### build docker image
```
img_name='train_test'
docker build -t ${img_name} -f Dockerfile .
```

### 构建docker 容器
```
cd /path/to/train_test
img_name='train_test'
cnt_name='train_test_cnt'
docker run --name ${cnt_name} -it --gpus all --ipc=host -v `pwd`:/workspace/train_test ${img_name}
```






