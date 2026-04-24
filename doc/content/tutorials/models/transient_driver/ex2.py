import torch
from matplotlib import pyplot as plt

# Load the result
res = torch.jit.load("result.pt")
I = dict(res.input.named_buffers())
O = dict(res.output.named_buffers())

# Strain and stress
nstep = 20
strain = [I["{}.strain".format(i)][0].item() for i in range(nstep)]
stress = [O["{}.stress".format(i)][0].item() for i in range(nstep)]

# Plot
plt.plot(strain, stress, "ko-")
plt.xlabel("Strain")
plt.ylabel("Stress")
plt.grid()
plt.tight_layout()
plt.savefig("curve.svg")
