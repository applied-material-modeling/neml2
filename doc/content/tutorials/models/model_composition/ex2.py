import neml2
from neml2.tensors import SR2
import torch

torch.set_default_dtype(torch.double)
factory = neml2.load_input("input.i")
eq1 = factory.get_model("eq1")
eq2 = factory.get_model("eq2")
eq3 = factory.get_model("eq3")

# Create the input variables
a = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03)
b = SR2.fill(100.0, 20.0, 10.0, 5.0, -30.0, -20.0)

# Evaluate the first model to get a_bar
a_bar = eq1.value({"a": a})["a_bar"]

# Evaluate the second model to get b_bar
b_bar = eq2.value({"b": b})["b_bar"]

# Evaluate the third model to get b_rate
eq3.w_0 = b_bar
eq3.w_1 = a_bar
b_rate = eq3.value({"a": a, "b": b})["b_rate"]

print("b_rate:")
print(b_rate)
