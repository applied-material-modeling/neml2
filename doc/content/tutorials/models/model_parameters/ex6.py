import neml2
from neml2.tensors import Scalar

model = neml2.load_model("input.i", "my_model")

# Before modification
print("G (before modification):")
print(model.G.tensor())

# After modification
model.G = Scalar.full(59000)
print("G (after modification):")
print(model.G.tensor())
