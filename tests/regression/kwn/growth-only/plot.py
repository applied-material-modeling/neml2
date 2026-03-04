import torch
import matplotlib.pyplot as plt
import numpy as np

data = torch.jit.load("result.pt")
res = dict(data.output.named_buffers())

xe = np.linspace(0,1,101)
xs = 0.5*(xe[:-1] + xe[1:])
s = 5.0
x = s*xs/(1-xs)

for i in range(1, 500):
    N = res[f"{i}.state/true_number_density"]
    print(N)
    print(N.shape)
    print(x.shape)
    plt.plot(x, N)
    plt.show()
