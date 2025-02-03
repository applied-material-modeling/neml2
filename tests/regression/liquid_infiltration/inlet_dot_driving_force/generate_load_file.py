#!/usr/bin/env python

# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch

##-----------------------  INPUT ----------------------- ##
# filename = "aLIndot.pt"
filename = "alpha.pt"
tensors_name = ["data", "time"]

cond = "user"  # Linspace or user
nsteps = 10000
in_start = 0
in_out = 108000 * 7
# torch.set_default_dtype(torch.float64)

userfname = "test.pt"
cond_drive = 0  # 1 for alpha, 0 for alphadot


def aSi_in_evolution_rate(t):
    if t <= 108000:
        # a_si_in = 0
        a_si_in = -(0.3 / 108000**2) * t**2 + 0.3
    elif t <= 108000 * 3:
        a_si_in = 3e4 / 108000
    else:
        a_si_in = 1e-4
    return a_si_in


def get_user_data(fname, drive_force):  # this is specifically for this problem

    dat = torch.jit.load(fname)
    full_dat = dat.state_dict()

    h = full_dat["output.1.state/h"]

    if drive_force == 0:
        alpha = full_dat["output.1.state/alpha"]
        dim2 = list(alpha.shape)
    else:
        aIn = full_dat["output.1.state/aLInDot"]
        dim2 = list(aIn.shape)

    delta = full_dat["output.1.state/delta"]
    dim2.insert(0, 1)

    tt = full_dat["input.1.forces/tt"]

    if drive_force == 0:
        aIn = full_dat["input.1.forces/aLInDot"]
    else:
        alpha = full_dat["input.1.forces/alpha"]

    dim = list(tt.shape)
    if dim == []:
        tt = tt.expand(dim2)
        aIn = aIn.expand(dim2)

    tt = torch.reshape(tt, dim2)
    aIn = torch.reshape(aIn, dim2)
    h = torch.reshape(h, dim2)
    alpha = torch.reshape(alpha, dim2)
    delta = torch.reshape(delta, dim2)

    for i in range(2, 1000000000000):
        basekey = "input." + str(i) + "."
        keyname = "input." + str(i) + ".forces/tt"
        if not (keyname in full_dat.keys()):
            break
        if dim == []:
            tt = torch.cat(
                [tt, torch.reshape(full_dat[basekey + "forces/tt"].expand(dim2), dim2)],
                dim=0,
            )
            if drive_force == 0:
                aIn = torch.cat(
                    [
                        aIn,
                        torch.reshape(full_dat[basekey + "forces/aLInDot"].expand(dim2), dim2),
                    ],
                    dim=0,
                )
            else:
                alpha = torch.cat(
                    [
                        alpha,
                        torch.reshape(full_dat[basekey + "forces/alpha"].expand(dim2), dim2),
                    ],
                    dim=0,
                )
        else:
            tt = torch.cat([tt, torch.reshape(full_dat[basekey + "forces/tt"], dim2)], dim=0)
            if drive_force == 0:
                aIn = torch.cat(
                    [aIn, torch.reshape(full_dat[basekey + "forces/aLInDot"], dim2)], dim=0
                )
            else:
                alpha = torch.cat(
                    [alpha, torch.reshape(full_dat[basekey + "forces/alpha"], dim2)], dim=0
                )

        h = torch.cat([h, torch.reshape(full_dat[basekey + "state/h"], dim2)], dim=0)

        if drive_force == 0:
            alpha = torch.cat(
                [alpha, torch.reshape(full_dat[basekey + "state/alpha"], dim2)], dim=0
            )
        else:
            aIn = torch.cat([aIn, torch.reshape(full_dat[basekey + "state/aLInDot"], dim2)], dim=0)

        delta = torch.cat([delta, torch.reshape(full_dat[basekey + "state/delta"], dim2)], dim=0)
    if drive_force == 0:
        return tt, alpha
    else:
        return tt, aIn


##-----------------------  RUN ----------------------- ##
# print(output)


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        if cond == "Linspace":
            output = torch.empty(nsteps)
            input = torch.linspace(in_start, in_out, nsteps)
            for iii in range(nsteps):
                output[..., iii] = aSi_in_evolution_rate(input[..., iii])
        elif cond == "user":
            input, output = get_user_data(userfname, cond_drive)
        else:
            ValueError("condition not yet implemented")

        # output[..., -1] = 1e-3

        self.register_buffer(tensors_name[0], output)
        self.register_buffer(tensors_name[1], input)
        # print(input)
        # print(output)


torch.jit.save(torch.jit.script(model()), filename)
