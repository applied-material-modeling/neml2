import torch
from matplotlib import pyplot as plt

# Load the saved model
result = torch.jit.load("result.pt")

I = dict(result.input.named_buffers())
O = dict(result.output.named_buffers())

expected = 14832

nstep = 100

T = [I["{}.forces/T".format(i)].item() for i in range(nstep)][1:]  # skip first step
t = [I["{}.forces/t".format(i)].item() for i in range(nstep)][1:]

q = [O["{}.state/q".format(i)].item() for i in range(nstep)][1:]
eta = [O["{}.state/eta".format(i)].item() for i in range(nstep)][1:]
phif_l = [O["{}.state/phif_l".format(i)].item() for i in range(nstep)][1:]
phif_s = [O["{}.state/phif_s".format(i)].item() for i in range(nstep)][1:]

# takes the integral of qdt using trapezoidal rule
heat_release = -0.5 * sum((t[i] - t[i + 1]) * (q[i + 1] + q[i]) for i in range(nstep - 2))

assert (
    abs(heat_release - expected) < 1
), f"Heat release {heat_release} does not match expected {expected}"

plt.plot(T, q, "ko-")
plt.xlabel("Temperature (K)")
plt.ylabel("Solidification fraction")
plt.grid()
plt.tight_layout()
plt.show()
