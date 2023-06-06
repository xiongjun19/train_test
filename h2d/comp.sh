nvcc profile.cu -o h2d_test
nvcc --default-stream per-thread profile.cu -o h2d_test
nvcc --default-stream per-thread profile_multi.cu -Xcompiler -fopenmp -o h2d_test
