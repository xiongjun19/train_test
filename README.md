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
以上命令生成并且进入了docker 容器

### 在容器里面编译与测试nccl_test
```
cd /workspace/train_test/nccl-tests
make 
./build/all_reduce_perf -b 1K -e 512M -f 2 -g 8 > nccl-test_log.txt
```

### 运行python 脚本检测torch all reduce 
cd /workspace/train_test/
OMP_NUM_THREADS=20 python -m  torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 test_nccl.py


### 运行训练测试样本
cd /workspace/train_test/train_test

#### 原始没有加profiler 的脚本
python -m torch.distributed.launch --nproc_per_node 8 --use_env nlp_example.py
#### 加上profiler
python -m torch.distributed.launch --nproc_per_node 8 --use_env nlp_example_prof.py


上面的代码会生成一个bert_large_log 的日志文件夹


