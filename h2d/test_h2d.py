import numpy as np
import torch
import time



shape=[4098, 256, 2048]
torch_tensor = torch.randn(shape, pin_memory=True)

iters=10
start = time.time()
for i in range(iters):
    device_mem = torch_tensor.cuda()
end = time.time() 
h2d_time = end - start
print(f"H2D time: {h2d_time*1e3} ms")


# Calculate bandwidth 
bandwidth = (np.prod(shape) * 4) / h2d_time / 1e6 * iters # MB/s

print(f"Bandwidth: {bandwidth} MB/s")
