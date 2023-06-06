

#include <thread>
#include <cstddef>
#include <stdlib.h>


__global__ void emptyKernel(float* d_data)
{
  // 不执行任何操作
}


void bandwidthTest(float* h_data, float* d_data, int size, cudaStream_t stream)
{
  // 将数据从主机复制到设备
  cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

  // 运行空的核函数
  emptyKernel<<<1, 1, 0, stream>>>(d_data);

  // 确保所有操作都完成
  cudaStreamSynchronize(stream);
}

int main(int argc, char* argv[])
{
  size_t size = static_cast<size_t>(1024) * 1024 * 1024; // 1GB

  // 分配主机内存
  float* h_data;
  cudaMallocHost((void**)&h_data, size);

  // 初始化数据
  for (size_t i = 0; i < size / sizeof(float); ++i)
  {
    h_data[i] = static_cast<float>(i);
  }

  //测试GPU卡的数量
  long device_num= strtol(argv[1], NULL, 10);

  // 创建 CUDA 流
  cudaStream_t streams[device_num];
  for(int i = 0; i < device_num; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // 分配设备内存
  float *d_data[device_num];
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    cudaMalloc((void**)&d_data[i], size);
  }

  // 创建 CUDA 事件来记录开始和结束时间
  cudaEvent_t start[device_num], stop[device_num];
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    cudaEventCreate(&start[i]);
    cudaEventCreate(&stop[i]);
  }

  // 预热 GPU
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    bandwidthTest(h_data, d_data[i], size, streams[i]);
  }

  // 等待所有预热操作完成
  for(int i = 0; i < device_num; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  // 清空 GPU 的缓存
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    cudaMemset(d_data[i], 0, size);
  }

  // 记录开始时间
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    cudaEventRecord(start[i], streams[i]);
  }

  // 运行带宽测试
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    bandwidthTest(h_data, d_data[i], size, streams[i]);
  }

  // 记录结束时间
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    cudaEventRecord(stop[i], streams[i]);
  }

  // 等待所有操作完成
  for(int i = 0; i < device_num; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  // 计算所用时间
  float milliseconds = 0;
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    float temp = 0;
    cudaEventElapsedTime(&temp, start[i], stop[i]);
    milliseconds = std::max(milliseconds, temp);
  }

  // 计算带宽（GB/s）
  float bandwidth = device_num * size / milliseconds / 1e6; // size * device_num (for eight GPUs)

  // 打印带宽
  printf("Total bandwidth: %.2f GB/s\n", bandwidth);

  // 销毁 CUDA 事件
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    cudaEventDestroy(start[i]);
    cudaEventDestroy(stop[i]);
  }

  // 释放内存
  cudaFreeHost(h_data);
  for(int i = 0; i < device_num; i++) {
    cudaSetDevice(i);
    cudaFree(d_data[i]);
  }

  // 销毁 CUDA 流
  for(int i = 0; i < device_num; i++) {
    cudaStreamDestroy(streams[i]);
  }

  return 0;
}

