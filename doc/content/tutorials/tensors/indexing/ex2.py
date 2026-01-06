import torch
from neml2.tensors import Tensor

torch.set_default_dtype(torch.double)

# Create a tensor with shape (; 5)
a = Tensor(torch.tensor([3, 4, 5, 6, 7]), 0)
print("a.base[2] =", a.base[2].item())
print("a.base[-1] =", a.base[-1].item())
print("a.base[-2] =", a.base[-2].item())
print()

# reate a tensor with shape (4;)
b = Tensor(torch.tensor([7, 6, 5, 4]), 1)
print("b.batch[2] =", b.batch[2].item())
print("b.batch[-1] =", b.batch[-1].item())
print("b.batch[-2] =", b.batch[-2].item())
