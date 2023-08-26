#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>
#include <chrono>


inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void profileCopies(float        *h_a, 
                   float        *h_b, 
                   float        *d, 
                   unsigned int  n,
                   char         *desc)
{
  printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  // events for timing
  cudaEvent_t startEvent, stopEvent; 

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}


int main(int argc, char* argv[])
{
  long nElements = strtol(argv[1], NULL, 10);
  const unsigned int bytes = nElements * sizeof(float);
  long deviceNum= strtol(argv[2], NULL, 10);
  std::cout << " input device Num is: " << deviceNum << std::endl;
  std::cout << " input bytes  is: " << bytes << std::endl;
  std::vector<float * > memVec;
  for(int i=0; i < deviceNum; ++i){
      float * hPinned;  
      cudaMallocHost((void**)&hPinned, bytes);
      memVec.push_back(hPinned);
      for(int j=0; j<nElements; ++j){
	      memVec[i][j] = i+j;
      }
  }
  // then init device Mem
  std::vector<float * > devVec;
  for(int i=0; i < deviceNum; ++i){
      cudaSetDevice(i);
      float * dMem;
      cudaMalloc((void**)&dMem, bytes);
      devVec.push_back(dMem);
  }
  


  std::chrono::steady_clock::time_point beg = std::chrono::steady_clock::now(); 
  #pragma omp parallel for num_threads(deviceNum)
  for(int i=0; i<deviceNum; ++i){
      cudaSetDevice(i);
      cudaMemcpy(devVec[i], memVec[i], bytes, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
  }
  std::chrono::steady_clock::time_point end  = std::chrono::steady_clock::now(); 
  auto timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
  auto bandWidth =  deviceNum * bytes * 1e-3 / timeDiff;
  std::cout << "time consumed: " << timeDiff << "\t Host to Device bandwidth (GB/s): " << bandWidth << std::endl;
  for(int i=0; i < deviceNum; ++i){
      cudaFree(devVec[i]);
      cudaFreeHost(memVec[i]);
  }

  return 0;
}
