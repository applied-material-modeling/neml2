import neml2
from neml2.tensors import Scalar, SR2
import torch

torch.set_default_dtype(torch.double)
model = neml2.load_model("input2.i", "model")

# Create input variables
# Unspecified variables are assumed to be zero
E = SR2.fill(0.01, 0.005, -0.001)
t = Scalar.full(1)

# Solve the implicit model
outputs = model.value({"forces/E": E, "forces/t": t})

# Get the solution
print("\nPlastic strain:")
print(outputs["state/Ep"])
