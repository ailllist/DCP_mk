import torch
import numpy as np

tensor1 = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
tensor2 = torch.tensor([[[[11, 21], [31, 41]], [[51, 61], [71, 81]]], [[[91, 0], [1, 2]], [[3, 4], [5, 6]]]])

tensor1 += tensor2
print(tensor1)