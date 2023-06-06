import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
import gc, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # normal 
def profile_copies(h_a, h_b, d, n, desc):
    print("\n{} transfers".format(desc))

    bytes = n * np.dtype(np.float32).itemsize

    # events for timing
    start_event = cuda.Event()
    stop_event = cuda.Event()

    start_event.record()
    cuda.memcpy_htod(d, h_a)
    stop_event.record()
    stop_event.synchronize()

    time_sec = start_event.time_till(stop_event) / 1000.0
    bandwidth = bytes / (time_sec * 1024 ** 3)
    print("  Host to Device bandwidth (GB/s): {}".format(bandwidth))

    start_event.record()
    cuda.memcpy_dtoh(h_b, d)
    stop_event.record()
    stop_event.synchronize()

    time_sec = start_event.time_till(stop_event) / 1000.0
    bandwidth = bytes / (time_sec * 1024 ** 3)
    print("  Device to Host bandwidth (GB/s): {}".format(bandwidth))

    if not np.allclose(h_a, h_b):
        print("*** {} transfers failed ***".format(desc))

def main():
    n_elements = 1600 * 1024 * 1024
    bytes = n_elements * np.dtype(np.float32).itemsize

    # host arrays
    h_a_pageable = np.arange(n_elements, dtype=np.float32)
    h_b_pageable = np.zeros(n_elements, dtype=np.float32)
    h_a_pinned = cuda.pagelocked_empty(n_elements, np.float32)
    h_b_pinned = cuda.pagelocked_empty(n_elements, np.float32)

    # device array
    d_a = cuda.mem_alloc(bytes)

    # initialize host arrays
    h_a_pinned[:] = h_a_pageable

    print("Device: {}".format(cuda.Device(0).name()))
    print("Transfer size (MB): {}".format(bytes / (1024 * 1024)))

    # perform copies and report bandwidth
    profile_copies(h_a_pageable, h_b_pageable, d_a, n_elements, "Pageable")
    profile_copies(h_a_pinned, h_b_pinned, d_a, n_elements, "Pinned")
    
    # cleanup
    d_a.free()

if __name__ == "__main__":
    main()
