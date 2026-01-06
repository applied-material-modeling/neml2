import torch
from neml2.tensors import SR2

A = SR2.zeros((5, 3), dtype=torch.float32, device=torch.device("cpu"))
