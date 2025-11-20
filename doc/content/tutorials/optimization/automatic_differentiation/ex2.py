import neml2
from neml2.tensors import SR2
import torch

torch.set_default_dtype(torch.double)
model = neml2.load_model("input.i", "model")

# Enable AD for both parameters
model.K.requires_grad_()
model.G.requires_grad_()

# Strain -> stress
strain = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)
stress = model.value({"forces/E": strain})["state/S"]

# Some additional operations on stress
A = stress.torch() ** 2 + 3.5 - 1
B = stress.torch() * strain.torch()
l = torch.linalg.norm(A + B)

# dl/dp
l.backward()
print("dl/dK =", model.K.grad.item())
print("dl/dG =", model.G.grad.item())
