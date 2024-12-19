import torch

##-----------------------  INPUT ----------------------- ##
filename = "aLIndot.pt"
tensors_name = ["data", "time"]

cond = "Linspace"
nsteps = 10000
in_start = 0
in_out = 108000 * 10
# torch.set_default_dtype(torch.float64)


def aSi_in_evolution_rate(t):
    if t <= 108000:
        a_si_in = 3e4 / 108000
    else:
        a_si_in = 0.0
    return a_si_in


##-----------------------  RUN ----------------------- ##
output = torch.empty(nsteps)


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        if cond == "Linspace":
            input = torch.linspace(in_start, in_out, nsteps)
            for iii in range(nsteps):
                output[..., iii] = aSi_in_evolution_rate(input[..., iii])
        else:
            ValueError("condition not yet implemented")

        # output[..., -1] = 1e-3

        self.register_buffer(tensors_name[0], output)
        self.register_buffer(tensors_name[1], input)
        # print(input)
        # print(output)


torch.jit.save(torch.jit.script(model()), filename)
