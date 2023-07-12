nvcc profile.cu -o h2d_test
nvcc --default-stream per-thread profile.cu -o h2d_test
nvcc --default-stream per-thread profile_multi.cu -Xcompiler -fopenmp -o h2d_test
nvcc -o bandtest bandwidthTest.cu -std=c++11
nvcc --default-stream per-thread -o bandtest bandwidthTest.cu -std=c++11
