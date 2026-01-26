import torch
import neml2
from neml2.tensors import SR2

torch.set_default_dtype(torch.double)

# Preparation
N = 10
device = torch.device("cpu")
model = neml2.load_model("input.i", "my_model")
model.to(device=device)

# Create the strain on the device
strain_min = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device=device)
strain_max = SR2.fill(0.5, 0.4, -0.2, 0.2, 0.3, 0.1, device=device)
strain = SR2.dynamic_linspace(strain_min, strain_max, N)

# Evaluate the model N times
for i in range(N):
    output = model.value({"forces/E": strain.dynamic[i]})
