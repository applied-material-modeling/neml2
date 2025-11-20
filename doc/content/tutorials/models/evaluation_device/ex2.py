import torch
import neml2
from neml2.tensors import SR2

torch.set_default_dtype(torch.double)
model = neml2.load_model("input.i", "my_model")

# Pick the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Send the model parameters to the device
model.to(device=device)

# Create the strain on the device
strain = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device=device)

# Evaluate the model
output = model.value({"forces/E": strain})

# Get the stress back to CPU
stress = output["state/S"].to(device=torch.device("cpu"))
