# coding=utf8


import torch
import numpy as np
import time


def test(device_num):
    last_dim = 1024 * device_num
    shape=[4098, 256, last_dim]
    torch_tensor = torch.randn(shape, pin_memory=True)
    device_list = list(range(device_num))

    iters=1
    start = time.time()
    for i in range(iters):
        torch.nn.parallel.scatter(torch_tensor, device_list, dim=-1)
    end = time.time() 
    h2d_time = end - start
    print(f"H2D time: {h2d_time*1e3} ms")
    # Calculate bandwidth 
    bandwidth = (np.prod(shape)  * 4) / h2d_time / 1e6 * iters # MB/s
    print(f"H2D time: {h2d_time*1e3} ms; Bandwidth: {bandwidth} MB/s")

if __name__ == '__main__':
    import sys
    device_num = int(sys.argv[1])
    test(device_num)

