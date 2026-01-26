import torch
from neml2.tensors import Tensor

torch.set_default_dtype(torch.double)

a0 = Tensor(torch.tensor([[0, 1], [2, 3], [4, 5]]), 1)
a1 = Tensor(torch.tensor([[3, 4], [5, 6], [7, 8]]), 1)
a = Tensor.dynamic_linspace(a0, a1, 4, 1)

print("a.batch[:2, -3:4] =")
print(a.batch[:2, -3:4])
