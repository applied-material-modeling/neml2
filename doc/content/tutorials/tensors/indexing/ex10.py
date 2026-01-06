import torch
from neml2.tensors import Tensor

torch.set_default_dtype(torch.double)

# Create a tensor with shape (5, 3, 1, 3; 1, 7, 8)
a = Tensor(torch.rand(5, 3, 1, 3, 1, 7, 8), 4)

# Batch indexing with ellipsis
print("a.batch[2:, ..., :].shape =", a.batch[2:, ..., :].shape)

# Batch indexing with none
print("a.batch[2:, :, :, None, :].shape =", a.batch[2:, :, :, None, :].shape)

# Batch indexing with both ellipsis and none
print("a.batch[..., None, :].shape =", a.batch[..., None, :].shape)

# Same rules apply to base indexing
print("a.base[..., None, :].shape =", a.base[..., None, :].shape)
