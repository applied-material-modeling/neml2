import torch
from neml2.tensors import Tensor

torch.set_default_dtype(torch.double)

# Create a tensor with shape (2, 2 ; 3, 1)
a = Tensor(
    torch.tensor([[[[1], [2], [3]], [[4], [5], [6]]], [[[-1], [-2], [-3]], [[-4], [-5], [-6]]]]), 2
)

# Single element indexing along batch dimensions
print("a.batch[1, 0] =")
print(a.batch[1, 0], "\n")
print("a.batch[0] =")
print(a.batch[0], "\n")

# Single element indexing along base dimensions
print("a.base[2, 0] =")
print(a.base[2, 0], "\n")
print("a.base[1] =")
print(a.base[1])
