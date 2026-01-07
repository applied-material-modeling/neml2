from neml2.tensors import Scalar
import torch

torch.set_default_dtype(torch.double)

# Create a tensor with shape (20;)
a0 = Scalar.full(0.0)
a1 = Scalar.full(19.0)
a = Scalar.dynamic_linspace(a0, a1, 20)

print("Basic syntax: start:stop:step")
print("a.batch[5:17:2] =")
print(a.batch[5:17:2], "\n")

print("Negative start and stop are counted backward")
print("a.batch[-15:-3:2] =")
print(a.batch[-15:-3:2], "\n")

print("start default to 0")
print("a.batch[:17:3] =")
print(a.batch[:17:3], "\n")

print("stop default to 'consuming all remaining elements'")
print("a.batch[12::2] =")
print(a.batch[12::2], "\n")

print("step default to 1")
print("a.batch[3:6:] =")
print(a.batch[3:6:], "\n")

print("Trailing colon(s) can be omitted")
print("a.batch[3:6] =")
print(a.batch[3:6], "\n")
print("a.batch[17:] =")
print(a.batch[17:], "\n")

print("The default is therefore equivalent to slicing the entire dimension")
print("a.batch[:] =")
print(a.batch[:])
