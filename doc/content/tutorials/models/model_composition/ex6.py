import neml2
from neml2.tensors import SR2
import torch

torch.set_default_dtype(torch.double)
eq = neml2.load_model("input_composed.i", "eq")

# Create the input variables
a = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)
b = SR2.fill(100.0, 20.0, 10.0, 5.0, -30.0, -20.0)

# Evaluate the composed model
b_rate = eq.value({"forces/a": a, "state/b": b})["state/b_rate"]

print("b_rate:")
print(b_rate)
