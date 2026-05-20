import torch

import neml2

torch.set_default_dtype(torch.double)
system = neml2.load_model("input1.i", "system")
print(system)
