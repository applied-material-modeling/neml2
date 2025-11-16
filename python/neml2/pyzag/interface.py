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

from pyzag import nonlinear

import torch
import neml2
from neml2.tensors import Tensor
from neml2.reserved import *


class NEML2PyzagModel(nonlinear.NonlinearRecursiveFunction):
    """Wraps a NEML2 model into a `nonlinear.NonlinearRecursiveFunction`

    Args:
        model (NEML2 model): the model to wrap

    Keyword Args:
        exclude_parameters (list of str): exclude these parameters from being wrapped as a pytorch parameter

    Additional args and kwargs are forwarded to NonlinearRecursiveFunction (and hence torch.nn.Module) verbatim
    """

    def __init__(
        self,
        model,
        *args,
        exclude_parameters=[],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model = model
        self._lookback = 1

        self._check_model()
        self._setup_parameters(exclude_parameters)
        self._setup_assemblers()

    @property
    def lookback(self):
        return self._lookback

    @lookback.setter
    def lookback(self, lookback):
        if lookback != 1:
            raise ValueError("NEML2 models only support lookback of 1")
        self._lookback = lookback

    @property
    def nstate(self):
        return self.model.input_axis().subaxis(STATE).size()

    @property
    def nforce(self):
        return self.model.input_axis().subaxis(FORCES).size()

    def _check_model(self):
        """Simple consistency checks, could be a debug check but we only call this once"""

        # First run diagnostics from NEML2
        # TODO: This check is temporarily disabled because the default diagnostics
        # from the C++ side disallow old variables to appear on the output axis.
        # However, one of the pyzag example models (km_mixed_model.i) relys on
        # MixedControlSetup to calculate old mixed conditions. To get rid of such
        # hack, some changes are required in pyzag.
        # neml2.diagnose(self.model)

        # Then pyzag specific checks
        input_axis = self.model.input_axis()
        output_axis = self.model.output_axis()

        # 1. Input axis should have state, old_state, forces, old_forces
        model_input_subaxes = input_axis.subaxis_names()
        expected_input_subaxes = [FORCES, OLD_FORCES, OLD_STATE, STATE]
        if model_input_subaxes != expected_input_subaxes:
            raise ValueError(
                "Wrapped NEML2 model should have {} as (the only) input subaxes. Got {}".format(
                    expected_input_subaxes, model_input_subaxes
                )
            )

        # 2. Output axis should just have the residual (and only the residual)
        model_output_subaxes = output_axis.subaxis_names()
        expected_output_subaxes = [RESIDUAL]
        if model_output_subaxes != expected_output_subaxes:
            raise ValueError(
                "Wrapped NEML2 model should have {} as (the only) output subaxes. Got {}".format(
                    expected_output_subaxes, model_output_subaxes
                )
            )

        # 3. All the variables on state should match the variables in the residual
        input_state_vars = input_axis.subaxis(STATE).variable_names()
        output_residual_vars = output_axis.subaxis(RESIDUAL).variable_names()
        if input_state_vars != output_residual_vars:
            raise ValueError(
                "Input state variables should match output residual variables. However, input state variables are {}, and output residual variables are {}".format(
                    input_state_vars, output_residual_vars
                )
            )

        # 4. Everything in old_state should be in state (but not the other way around)
        input_old_state_vars = input_axis.subaxis(OLD_STATE).variable_names()
        if not set(input_old_state_vars) <= set(input_state_vars):
            raise ValueError(
                "Input old state variables should be a subset of input state variables. However, input state variables are {}, and input old state variables are {}".format(
                    input_state_vars, input_old_state_vars
                )
            )

        # 5. Everything in old_forces should be in forces (but not the other way around)
        input_forces_vars = input_axis.subaxis(FORCES).variable_names()
        input_old_forces_vars = input_axis.subaxis(OLD_FORCES).variable_names()
        if not set(input_old_forces_vars) <= set(input_forces_vars):
            raise ValueError(
                "Input old forces should be a subset of input forces. However, input forces are {}, and input old forces are {}".format(
                    input_forces_vars, input_old_forces_vars
                )
            )

    def _setup_parameters(self, exclude_parameters):
        """Mirror parameters of the NEML2 model with torch.nn.Parameter

        Args:
            exclude_parameters (list of str): NEML2 parameters to exclude
        """
        self.parameter_names = []
        for pname, param in self.model.named_parameters().items():
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
            current_value = self.model.get_parameter(pname).tensor()
            # We may need to update the batch shape
            batch_dim = new_value.dim() - current_value.base.dim()
            self.model.set_parameter(pname, Tensor(new_value.clone(), batch_dim))

    def _setup_assemblers(self):
        """Setup the assemblers for the state and forces"""

        input_axis = self.model.input_axis()
        output_axis = self.model.output_axis()

        self.input_asm = neml2.VectorAssembler(input_axis)
        self.output_asm = neml2.VectorAssembler(output_axis)
        self.deriv_asm = neml2.MatrixAssembler(output_axis, input_axis.subaxis(STATE))
        self.old_deriv_asm = neml2.MatrixAssembler(output_axis, input_axis.subaxis(OLD_STATE))

        self.state_asm = neml2.VectorAssembler(input_axis.subaxis(STATE))
        self.forces_asm = neml2.VectorAssembler(input_axis.subaxis(FORCES))

    def _disassemble_input(self, state, forces):
        """Assemble the model input from the flat tensors

        Args:
            state (torch.Tensor): tensor containing the model state
            forces (torch.Tensor): tensor containing the model forces
        """
        # State and forces should have the same batch shape
        assert state.shape[:-1] == forces.shape[:-1]
        batch_shape = (state.shape[0] - self.lookback,) + state.shape[1:-1]
        bdim = len(batch_shape)

        # Disassemble state and old_state
        state_vars = self.state_asm.split_by_variable(Tensor(state, bdim))
        new_state_vars = {k: v.dynamic[self.lookback :] for k, v in state_vars.items()}
        old_state_vars = {k.old(): v.dynamic[: -self.lookback] for k, v in state_vars.items()}

        # Disassemble forces and old_forces
        forces_vars = self.forces_asm.split_by_variable(Tensor(forces, bdim))
        new_forces_vars = {k: v.dynamic[self.lookback :] for k, v in forces_vars.items()}
        old_forces_vars = {k.old(): v.dynamic[: -self.lookback] for k, v in forces_vars.items()}

        return new_state_vars | old_state_vars | new_forces_vars | old_forces_vars

    def _adapt_for_pyzag(self, r, J, J_old):
        """Adapt the residual and Jacobians for pyzag

        pyzag has additional requirements on residual and Jacobians:
          1. The residual and Jacobians should have the same batch shape
          2. The Jacobians should be square

        Args:
            r (neml2.Tensor): residual
            J (neml2.Tensor): Jacobian
            J_old (neml2.Tensor): Jacobian for the old state

        Returns:
            tuple of torch.Tensor: residual, Jacobian
        """
        # Make the residual and Jacobian have the same batch shape
        J = J.batch.expand(r.batch.shape)
        J_old = J_old.batch.expand(r.batch.shape)

        # Make the Jacobian square
        # The current Jacobian should already be square
        assert J.base.shape[-1] == J.base.shape[-2]

        # The old Jacobian may not be, in which case we will rely on the deriv_asm to make it square
        if J_old.base.shape[-1] != J_old.base.shape[-2]:
            J_old_vars = self.old_deriv_asm.split_by_variable(J_old)
            J_old_vars_adapted = {}
            for yvar, vs in J_old_vars.items():
                J_old_vars_adapted[yvar] = {k.current(): v for k, v in vs.items()}
            J_old = self.deriv_asm.assemble_by_variable(J_old_vars_adapted)
        assert J_old.base.shape[-1] == J_old.base.shape[-2]

        return r.torch(), torch.stack([J_old.torch(), J.torch()])

    def forward(self, state, forces):
        """Actually call the NEML2 model and return the residual and Jacobian

        Args:
            state (torch.tensor): tensor with the flattened state
            forces (torch.tensor): tensor with the flattened forces
        """
        # Update the parameter values
        self._update_parameter_values()

        # Make a big LabeledVector with the input
        x = self._disassemble_input(state, forces)

        # Call the model
        r, J = self.model.value_and_dvalue(x)

        # Assemble residual
        r_vec = self.output_asm.assemble_by_variable(r)

        # Assemble Jacobian
        J_new_mat = self.deriv_asm.assemble_by_variable(J)
        J_old_mat = self.old_deriv_asm.assemble_by_variable(J)

        # At this point, the residual and Jacobians should be good to go
        return self._adapt_for_pyzag(r_vec, J_new_mat, J_old_mat)
