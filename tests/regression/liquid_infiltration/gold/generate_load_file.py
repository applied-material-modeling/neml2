## Copyright 2024, UChicago Argonne, LLC
## All Rights Reserved
## Software Name: NEML2 -- the New Engineering material Model Library, version 2
## By: Argonne National Laboratory
## OPEN SOURCE LICENSE (MIT)
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.

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
