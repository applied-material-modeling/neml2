import torch
from matplotlib import pyplot as plt

# Load the result
res = torch.jit.load("result.pt")
O = dict(res.output.named_buffers())

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
nstep = 100
nvel = 2  # two launching velocities
nproj = 5  # five projectiles (with different dynamic viscosity)
for i in range(nvel):
    for j in range(nproj):
        x = [O["{}.state/x".format(n)][i, j, 0].item() for n in range(1, nstep)]
        y = [O["{}.state/x".format(n)][i, j, 1].item() for n in range(1, nstep)]
        ax[i].plot(x, y, "--")
    ax[i].set_xlabel("x")
    ax[i].set_ylabel("y")
    ax[i].grid()
fig.tight_layout()
fig.savefig("trajectories.svg")
