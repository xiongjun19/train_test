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


### H2D 测试 
```
cd /workspace/train_test/h2d
nvcc --default-stream per-thread profile_multi.cu -Xcompiler -fopenmp -o h2d_test
CUDA_VISIBLE_DEVICES=0,1 ./h2d_test 1   # this using one card h2d
CUDA_VISIBLE_DEVICES=0,1 ./h2d_test 2   # this testing the two cards h2d 
```

### 在容器里面编译与测试nccl_test
```
cd /workspace/train_test/nccl-tests
make 
./build/all_reduce_perf -b 1K -e 512M -f 2 -g 8 > nccl-test_log.txt
CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -b 1K -e 128M -f 2 -g 2 > nccl-test_log_0_1.txt
CUDA_VISIBLE_DEVICES=0,2 ./build/all_reduce_perf -b 1K -e 128M -f 2 -g 2 > nccl-test_log_0_2.txt
CUDA_VISIBLE_DEVICES=0,5 ./build/all_reduce_perf -b 1K -e 128M -f 2 -g 2 > nccl-test_log_0_5.txt
```

#### 卡和卡之间两两批量测试
```
cd /workspace/train_test/nccl-tests
python two_cards_batch_test.py -n 10 -o two_out_dir   # 这里的10 表示10张卡
python parser -i two_out_dir -o two_card_bw.csv
```

### peer2peer 测试 
在容器中执行以下命令可以进行peer2peer 的测试
step1 编译
```
cd /workspace/train_test/cuda_examples/0_Simple/simpleP2P
bash comp.sh
```
step2 手动运行peer2peer copy 
```
./test_run
CUDA_VISIBLE_DEVICES=1,5 ./test_run
```

卡和卡之间两两批量运行peer2peer copy 
```
cd /workspace/train_test/cuda_examples/0_Simple/simpleP2P
python two_cards_batch_test.py -o p2p_out -n 10
python parser.py -i p2p_out/ -o batch_res.csv
```


### 运行训练测试样本
```
cd /workspace/train_test/train_test
```

#### 原始没有加profiler 的脚本
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env nlp_example.py
```
#### 加上pytorch profiler
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env nlp_example_prof.py
```
上面的代码会生成一个bert_large_log 的日志文件夹

#### 利用nsys 得到日志(maybe better than pytorch profiler)
```
nsys profile  -c cudaProfilerApi -f true --stats true  -o bert_large_nsys.qdrep python -m torch.distributed.launch --nproc_per_node 8 --use_env nlp_example_nsys.py
```
上面的脚本会生成bert_large_nsys 开头的问题件， 这些日志文件比较重要

