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

import math
import typing

from pyzag import nonlinear

import torch
import neml2
from neml2.tensors import Tensor
from neml2.es import AssembledVector, AssembledMatrix, AxisLayout, SparseMatrix
from neml2.reserved import FORCES, OLD_FORCES, OLD_STATE, STATE


class NEML2PyzagModel(nonlinear.NonlinearRecursiveFunction):
    """Wraps a NEML2 nonlinear system into a `nonlinear.NonlinearRecursiveFunction`

    Args:
        sys: the NEML2 nonlinear system to wrap

    Keyword Args:
        exclude_parameters (list of str): exclude these parameters from being wrapped as a pytorch parameter

    Additional args and kwargs are forwarded to NonlinearRecursiveFunction (and hence torch.nn.Module) verbatim
    """

    def __init__(
        self,
        sys: neml2.NonlinearSystem,
        *args,
        exclude_parameters: list[str] = [],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(sys, neml2.NonlinearSystem):
            raise TypeError(
                f"sys should be a neml2.NonlinearSystem, instead got {type(sys)}. Please use neml2.load_nonlinear_system or neml2.Factory.get_nonlinear_system to load a nonlinear system from the input file."
            )

        self.sys = sys
        self.model = sys.model()
        self._lookback = 1

        self._setup_maps()
        self._check_model()
        self._setup_parameters(exclude_parameters)

    @property
    def lookback(self) -> int:
        return self._lookback

    @lookback.setter
    def lookback(self, lookback: int):
        if lookback != 1:
            raise ValueError("NEML2 models only support lookback of 1")
        self._lookback = lookback

    @property
    def nstate(self) -> int:
        return sum([math.prod(self.slayout.base_sizes(i)) for i in range(self.slayout.nvar())])

    @property
    def nforce(self) -> int:
        return sum([math.prod(self.flayout.base_sizes(i)) for i in range(self.flayout.nvar())])

    def forward(
        self, state: torch.Tensor, forces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Actually call the NEML2 model and return the residual and Jacobian

        Args:
            state (torch.Tensor): tensor with the flattened state
            forces (torch.Tensor): tensor with the flattened forces
        """
        # Update the parameter values
        self._update_parameter_values()

        # Update the current state of the nonlinear system
        self._update_sys(state, forces)

        # Call the model
        # A := dr/ds
        # B := dr/dg, where g includes s_n, f, and f_n
        # b := -r
        A, B, b = self.sys.A_and_B_and_b()

        # Pyzag needs
        # r := -b
        # J := dr/ds
        # Jn := dr/ds_n
        return self._adapt_for_pyzag(A, B, b)

    def _check_model(self):
        """Simple consistency checks, could be a debug check but we only call this once"""

        # First run diagnostics from NEML2
        # TODO: This check is temporarily disabled because the default diagnostics
        # from the C++ side disallow old variables to appear on the output axis.
        # However, one of the pyzag example models (km_mixed_model.i) relys on
        # MixedControlSetup to calculate old mixed conditions. To get rid of such
        # hack, some changes are required in pyzag.
        # neml2.diagnose(self.model)

        # Helper function to replace the variable prefix
        def _replace_prefix(vars: list[str], new_prefix: str) -> list[str]:
            return [f"{new_prefix}/{v.split('/', 1)[1]}" for v in vars]

        # Every old variable (state or force) should have a corresponding (current) variable (but not the other way around)
        if not set(_replace_prefix(self.snlayout.vars(), STATE)) <= set(self.slayout.vars()):
            raise ValueError(
                "Input old state variables should be a subset of input state variables. However, input state variables are {}, and input old state variables are {}".format(
                    self.slayout.vars(), self.snlayout.vars()
                )
            )
        if not set(_replace_prefix(self.fnlayout.vars(), FORCES)) <= set(self.flayout.vars()):
            raise ValueError(
                "Input old force variables should be a subset of input force variables. However, input force variables are {}, and input old force variables are {}".format(
                    self.flayout.vars(), self.fnlayout.vars()
                )
            )

    def _setup_parameters(self, exclude_parameters: list[str]):
        """Mirror parameters of the NEML2 model with torch.nn.Parameter

        Args:
            exclude_parameters (list of str): NEML2 parameters to exclude
        """
        self.parameter_names = []
        for pname, param in self.sys.named_parameters().items():
            if "." in pname:
                errmsg = "Parameter name {} contains a period, which is not allowed. \nMake sure Settings/parameter_name_separator='_' in the NEML2 input file."
                raise ValueError(errmsg.format(pname))
            if pname in exclude_parameters:
                continue
            # We need to separately track the parameter names because of the reparameterization system
            # What torch does when it reparamterizes models is replace the variable with a lambda method that calls the scaling function
            # over some new, reparamterized, variable.  If we rely on using torch.named_parameters() we will get the new "unscaled" variable, rather
            # than the scaled value we want to pass to the model.  Caching the names prevents this.
            self.parameter_names.append(pname)
            param.requires_grad_(True)
            self.register_parameter(pname, torch.nn.Parameter(param.torch()))

    def _update_parameter_values(self):
        """Copy over new parameter values"""
        for pname in self.parameter_names:
            # See comment in _setup_parameters, using getattr here lets us reparamterize things with torch
            new_value = getattr(self, pname)
            current_value = self.sys.get_parameter(pname).tensor()
            # We may need to update the batch shape
            dynamic_dim = new_value.dim() - current_value.base.dim() - current_value.intmd.dim()
            self.sys.set_parameter(
                pname, Tensor(new_value.clone(), dynamic_dim, current_value.intmd.dim())
            )

    def _setup_maps(self):
        """
        Setup the maps for assembly purposes

        In the C++ backend, the nonlinear system distinguishes between unknowns (u) and given variables (g).
        However, pyzag expects contiguous-in-time representation of the state and forces, where unknowns and old
        unknowns come from the state tensor, and forces and old forces come from the forces tensor.

        Note that the given variables include old unknowns, forces, and old forces. So we need to setup maps from
        the pyzag representation to the nonlinear system representation for both unknowns and given variables.
        """

        # Helper function to extract a sublayout given a prefix
        def _extract_sublayout(
            layout: AxisLayout, prefix: str, new_prefix: typing.Union[str, None] = None
        ) -> AxisLayout:
            subvars = []
            intmd_shapes = []
            base_shapes = []
            for i in range(layout.nvar()):
                v = layout.var(i)
                if v.startswith(prefix):
                    if not new_prefix:
                        subvars.append(v)
                    else:
                        subvars.append(f"{new_prefix}/{v.split('/', 1)[1]}")
                    intmd_shapes.append(layout.intmd_sizes(i))
                    base_shapes.append(layout.base_sizes(i))
            return AxisLayout([subvars], intmd_shapes, base_shapes, [AxisLayout.IStructure.DENSE])

        # given variables (from neml2, includes old unknowns, forces, and old forces))
        glayout = self.sys.glayout()
        gvars = glayout.vars()

        # setup layouts for assembly purposes
        self.rlayout = self.sys.blayout()
        self.slayout = self.sys.ulayout()
        self.snlayout = _extract_sublayout(self.slayout, STATE, OLD_STATE)
        self.flayout = _extract_sublayout(glayout, FORCES)
        self.fnlayout = _extract_sublayout(glayout, FORCES, OLD_FORCES)

        # state variables (unknowns)
        self.svars = self.slayout.vars()

        # forces
        self.fvars = [v for v in gvars if v.startswith(FORCES)]

        # figure out how gvars map to snvars
        self._sn_to_g_map = [-1] * self.snlayout.nvar()
        for i, snv in enumerate(self.snlayout.vars()):
            for j, gv in enumerate(gvars):
                if snv == gv:
                    self._sn_to_g_map[i] = j
                    break

    def _update_sys(self, state: torch.Tensor, forces: torch.Tensor):
        """
        Disassemble the model input forces, old forces, and old state from
        the flat tensors and update them in the nonlinear system

        Args:
            state (torch.Tensor): tensor containing the model state
            forces (torch.Tensor): tensor containing the model forces
        """
        # State and forces should have the same batch shape
        assert state.shape[:-1] == forces.shape[:-1]
        batch_shape = (state.shape[0] - self.lookback,) + state.shape[1:-1]
        bdim = len(batch_shape)

        # Convert the input tensors to NEML2 tensors
        sf = Tensor(state, bdim)
        ff = Tensor(forces, bdim)

        # Update current state
        sf_np1 = AssembledVector(self.slayout, [sf.dynamic[self.lookback :]])
        self.sys.set_u(sf_np1)

        # Update old state, forces, and old forces
        sf_n = AssembledVector(self.snlayout, [sf.dynamic[: -self.lookback]])
        ff_np1 = AssembledVector(self.flayout, [ff.dynamic[self.lookback :]])
        ff_n = AssembledVector(self.fnlayout, [ff.dynamic[: -self.lookback]])
        self.sys.set_g(sf_n)
        self.sys.set_g(ff_np1)
        self.sys.set_g(ff_n)

    def _adapt_for_pyzag(
        self, A: AssembledMatrix, B: AssembledMatrix, b: AssembledVector
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the residual and Jacobians from A, B, and b

        Args:
            A (AssembledMatrix): Jacobians w.r.t. unknowns
            B (AssembledMatrix): Jacobians w.r.t. given variables
            b (AssembledVector): (negative) residuals

        Returns:
            tuple of neml2.Tensor: residual, Jacobian w.r.t. unknowns, Jacobian w.r.t. old state
        """

        r = -b.tensors[0]
        J = A.tensors[0][0]

        # We need to extract the Jacobian w.r.t. old state from B
        B_sp = B.disassemble().tensors
        Jn_sp = [[Tensor()] * self.snlayout.nvar() for _ in range(self.rlayout.nvar())]
        for i in range(self.rlayout.nvar()):
            for j in range(self.snlayout.nvar()):
                g_idx = self._sn_to_g_map[j]
                if g_idx != -1:
                    Jn_sp[i][j] = B_sp[i][g_idx]
        Jn = SparseMatrix(self.rlayout, self.snlayout, Jn_sp).assemble().tensors[0][0]

        # Make the residual and Jacobian have the same batch shape
        J = J.dynamic.expand(r.dynamic.shape)
        Jn = Jn.dynamic.expand(r.dynamic.shape)

        # Make sure the Jacobians are square
        assert J.base.shape[-1] == J.base.shape[-2]
        assert Jn.base.shape[-1] == Jn.base.shape[-2]

        return r.torch(), torch.stack([Jn.torch(), J.torch()])
