import torch
import numpy as np

arr1 = np.array([[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]], np.float32)
tensor1 = torch.tensor(arr1)
print(torch.softmax(tensor1, dim=2))