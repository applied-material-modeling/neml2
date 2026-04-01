import torch
import matplotlib.pyplot as plt
import numpy as np

data = torch.jit.load("result.pt")
res = dict(data.output.named_buffers())

time = np.logspace(-4,1,500)

nsteps = 500

nbins = 300

xe = np.linspace(0,0.5,nbins+1)
x = 0.5*(xe[:-1] + xe[1:])

N = torch.stack([res[f"{i}.state/number_density"] for i in range(1,nsteps)])
vf = torch.stack([res[f"{i}.state/vf"] for i in range(1,nsteps)])
X = torch.stack([res[f"{i}.state/x_Cu"] for i in range(1,nsteps)])

plt.figure()
plt.plot(time[1:], vf)
plt.xlabel("Time (hr)")
plt.ylabel("$f$ $\mathrm{AlCu}_\mathrm{2}$")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(time[1:], X)
plt.xlabel("Time (hr)")
plt.ylabel("$X$ Cu")
plt.tight_layout()
plt.show()

plt.figure()
for i in range(0,nsteps-1,50):
    plt.semilogx(x, N[i], label = f"{time[i]:.1e} hr")

plt.xlabel("$R$ ($\mathrm{\mu}$m)")
plt.ylabel("$N$ (per $\mathrm{\mu m}^3$)")
plt.legend(loc='best')
plt.tight_layout()
plt.show()
