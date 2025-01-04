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


class Surrogate(torch.nn.Module):
    """
    This model defines an arbitrary power-law plastic flow rate with two
    additional internal variables. The internal variables (G and C) are made up
    and do not necessarily have any physical meaning.

    Most importantly, this model can represent a feed-forward neural network
    without loss of generality:
      - It has a bunch of parameters.
      - It has a (nonlinear) forward operator that maps some input to
        some output.
      - The shapes of input and output are not necessarily the same.
    """

    def __init__(self, dtype):
        super().__init__()
        self.sy = torch.nn.Parameter(torch.tensor(1000.0, dtype=dtype), requires_grad=False)
        self.eta = torch.nn.Parameter(torch.tensor(10.0, dtype=dtype), requires_grad=False)
        self.n = torch.nn.Parameter(torch.tensor(3.0, dtype=dtype), requires_grad=False)
        self.Q = torch.nn.Parameter(torch.tensor(5e4, dtype=dtype), requires_grad=False)
        self.R = torch.nn.Parameter(torch.tensor(8.3145, dtype=dtype), requires_grad=False)
        self.G0 = torch.nn.Parameter(torch.tensor(3e-3, dtype=dtype), requires_grad=False)
        self.C0 = torch.nn.Parameter(torch.tensor(4e-3, dtype=dtype), requires_grad=False)

    def forward(self, s, T, G, C):
        """
        The forward operator maps 4 inputs to 3 outputs.

        The inputs are
           s: von Mises stress
           T: temperature
           G: grain size
           C: stoichiometry

        The outputs are
           ep_dot: rate of the equivalent plastic strain
            G_dot: rate of grain growth
            C_dot: reaction rate
        """
        f = s - self.sy
        Hf = (torch.sign(f) + 1.0) / 2.0
        rep = (torch.abs(f) / self.eta) ** self.n * Hf
        rG = torch.exp(-G / self.G0)
        rC = torch.exp(-C / self.C0)
        aT = torch.exp(-self.Q / self.R / T)

        ep_dot = aT * (rG + rC) * rep

        return ep_dot


if __name__ == "__main__":
    # Instantiate the model and trace it into a torch script.
    # See the following pages for a gentle introduction to torch script:
    #   https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
    #   https://pytorch.org/docs/stable/jit.html
    surrogate = torch.jit.script(Surrogate(torch.float64))

    # Write the torch script to disk. NEML2 can read this torch script as part
    # of the material model.
    torch.jit.save(surrogate, "surrogate.pt")
