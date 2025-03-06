@insert-title:tutorials-optimization-parameter-calibration

[TOC]

## Problem definition

Suppose we are given the experimental data of one pair of strain \f$\boldsymbol{\varepsilon}^*\f$ and stress \f$\boldsymbol{\sigma}^*\f$ measurement of a specimen, and we are tasked to infer the bulk \f$K\f$ and shear \f$G\f$ modulus of the material (assuming it's homogeneous, isotropic, purely elastic, and small strain assumptions hold).

Let us define the loss function as the *distance* between the stress prediction and the stress measurement, i.e.
\f[
  l(K, G) = \norm{\boldsymbol{\sigma}(\boldsymbol{\varepsilon}^*; K, G) - \boldsymbol{\sigma}^*}^2.
\f]
The material parameters, \f$K^*\f$ and \f$G^*\f$, can then be "inferred" by minimizing the loss function
\f[
  \left( K^*, G^* \right) = \mathop{\mathrm{argmin}}\limits_{K, G} l.
\f]

## Calibration

The following Python script uses the simple gradient descent algorithm to find \f$K^*\f$ and \f$G^*\f$:
\f{align*}
  K_{i+1} = K_i - \gamma \pdv{l}{K_i},
  G_{i+1} = G_i - \gamma \pdv{l}{G_i},
\f}
which iterates until the loss function \f$l\f$ is sufficiently small. A constant learning rate \f$\gamma = 1e-3\f$ is chosen in this example.

@source:src1
```python
import neml2
from neml2.tensors import SR2
import torch
from matplotlib import pyplot as plt

model = neml2.load_model("input.i", "model")

# Experimental data
strain_exp = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)
stress_exp = SR2.fill(2.616, 1.836, 0.588, 0.312, 0.9361, 0.468)

# The gradient descent loop
gamma = 1
loss_history = []
K_history = []
G_history = []
for i in range(100):
    # Enable AD
    model.K.requires_grad_()
    model.G.requires_grad_()

    # Calculate loss
    stress = model.value({"forces/E": strain_exp})["state/S"]
    l = torch.linalg.norm(stress.torch() - stress_exp.torch()) ** 2

    # Record state
    loss_history.append(l.item())
    K_history.append(model.K.torch().item())
    G_history.append(model.G.torch().item())

    # Update parameters
    l.backward()
    model.K = model.K.tensor().detach() - gamma * model.K.grad
    model.G = model.G.tensor().detach() - gamma * model.G.grad

# Make plots
fig, axl = plt.subplots(figsize=(8, 5))
axl.plot(torch.arange(100), loss_history, "k-")
axl.set_xscale("log")
axl.set_yscale("log")
axl.set_xlabel("Iteration")
axl.set_ylabel("Loss")

axp = axl.twinx()
axp.plot(torch.arange(100), K_history, "r--", label="Bulk modulus")
axp.plot(torch.arange(100), G_history, "b--", label="Shear modulus")
axp.set_ylabel("Parameter value")
axp.legend()

fig.tight_layout()
fig.savefig("calibration.svg")
```
@endsource

![Material parameter calibration](tutorials/optimization/parameter_calibration/calibration.svg){html: width=85%}

@insert-page-navigation
