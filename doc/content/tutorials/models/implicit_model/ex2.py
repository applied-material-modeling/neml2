import neml2
import torch

torch.set_default_dtype(torch.double)
system = neml2.load_model("input1.i", "system")
print(system)
