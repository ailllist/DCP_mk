import torch

mat1 = torch.tensor([[[1, 2, 5], [3, 4, 7]], [[5, 6, 0], [7, 8, 1]], [[9, 10, 11], [12, 13, 14]], [[9, 10, 11], [12, 13, 14]]])
print(mat1.shape)
mat1 = mat1.transpose(-1, -2)
print(mat1.shape)
