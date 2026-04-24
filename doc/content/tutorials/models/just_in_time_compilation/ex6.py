import neml2
from neml2.tensors import Scalar
import torch

torch.set_default_dtype(torch.double)
model = neml2.load_model("input.i", "eq")

# Create example input variables for tracing
a = Scalar.full(1.0)
b = Scalar.full(2.0)
c = Scalar.full(3.0)
t = Scalar.full(0.1)
a_n = Scalar.full(0.0)
b_n = Scalar.full(1.0)
c_n = Scalar.full(2.0)
t_n = Scalar.full(0.0)

# Evaluate the model for the first time
# This is when tracing takes place
inputs = {
    "a": a,
    "b": b,
    "c": c,
    "t": t,
    "a~1": a_n,
    "b~1": b_n,
    "c~1": c_n,
    "t~1": t_n,
}
for i in range(10):
    model.value(inputs)

print(torch.jit.last_executed_optimized_graph())  # type: ignore[attr-defined]
