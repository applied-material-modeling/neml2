@insert-title:tutorials-optimization-automatic-differentiation

[TOC]

## Model definition

For demonstration purposes, let us go back to an earlier example model -- linear isotropic elasticity:
```
[Models]
  [model]
    type = LinearIsotropicElasticity
    strain = 'forces/E'
    stress = 'state/S'
    coefficient_types = 'BULK_MODULUS SHEAR_MODULUS'
    coefficients = '1.4e5 7.8e4'
  []
[]
```

Recall that both the bulk modulus and the shear modulus are treated as *trainable* parameters, which can be verified by inspecting the model summary:
@source:src1
```python
import neml2

model = neml2.load_model("input.i", "model")
print(model)
```
@endsource

Output:
```
@attach-output:src1
```

## Automatic differentiation

NEML2 tensors can be used interchangeably with PyTorch tensors in a seamless fashion. Function graph traced through NEML2 tensor operations can be back-propagated using PyTorch's autograd engine.

For example, the following Python script demonstrates the calculation of \f$\pdv{l}{p}\f$ using an arbitrary, scalar-valued loss function \f$l\f$ obtained by a combination of NEML2 operations and PyTorch operations.

@source:src2
```python
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
```
@endsource

Output:
```
@attach-output:src2
```

@insert-page-navigation
