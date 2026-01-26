import neml2
from neml2.tensors import SR2
import torch
from matplotlib import pyplot as plt

torch.set_default_dtype(torch.double)
model = neml2.load_model("input.i", "model")

# Experimental data
# Spoiler: These data correspond to K = 14 and G = 7.8
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
