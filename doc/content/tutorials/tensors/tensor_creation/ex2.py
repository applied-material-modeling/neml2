import torch
from neml2.tensors import Tensor

# A raw torch tensor
A = torch.rand(5, 3, 2, 8)
print("Shape of A:", A.shape, "\n")

# Mark it with batch dimension of 2
B = Tensor(A, 2)
print("      Shape of B:", B.shape)
print("Batch shape of B:", B.batch.shape)
print(" Base shape of B:", B.base.shape)
