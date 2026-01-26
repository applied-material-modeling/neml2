import neml2
from neml2.tensors import SR2
import torch

torch.set_default_dtype(torch.double)
model = neml2.load_model("input.i", "my_model")

# Create the strain
strain = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)

# Evaluate the model
output = model.value({"forces/E": strain})

# Get the stress
stress = output["state/S"]

print("strain:")
print(strain)
print("stress:")
print(stress)
