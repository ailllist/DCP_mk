import torch
import numpy as np

a = torch.nn.LeakyReLU(negative_slope=0)
arr1 = np.array([-10], np.float32)
tmp = torch.tensor(arr1)
tmp = tmp.to("cuda")
print(a(tmp))