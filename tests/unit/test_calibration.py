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

"""calibration story (D-038, D-039, D-040): eager-mode autograd through
Python-native models with ``register_typed_parameter`` calibration parameters.

Covers:

* parameter visibility (``model.parameters()`` returns the registered params).
* autograd through a leaf — closed-form `dstress/dE`, `dstress/dnu` agreement.
* autograd through a :class:`~neml2.models.common.ComposedModel` chain.
* end-to-end calibration fit: recover E from synthetic stress data using
  L-BFGS, well within 1% of ground truth in a few iterations.

The AOTI path (``aoti_mode = true``) freezes parameters to constants at
export time and is exercised by ``tests/aoti/test_AOTIMode.cxx``; this
file pins only the *eager* autograd behaviour.

Autograd through ``ImplicitUpdate`` (the IFT backward) was exercised here via a
full J2 return-map; that integration coverage now lives with the migrated
``TransientDriver`` / ``TransientRegression`` scenarios.
"""

import math

import torch

from neml2 import load_string
from neml2.models.common import ComposedModel
from neml2.models.solid_mechanics.elasticity import LinearIsotropicElasticity
from neml2.types import SR2

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Parameter visibility
# ---------------------------------------------------------------------------


def test_linear_isotropic_elasticity_params_are_visible():
    model = LinearIsotropicElasticity(E=100.0, nu=0.3)
    params = dict(model.named_parameters())
    assert set(params) == {"E", "nu"}
    assert all(p.requires_grad for p in params.values())


def test_composed_model_aggregates_inner_params():
    # LinearIsotropicHardening is schema-only (no Python constructor) — build via HIT.
    hard = load_string(
        "[Models]\n  [h]\n    type = LinearIsotropicHardening\n"
        "    hardening_modulus = 1000.0\n  []\n[]"
    ).get_model("h")
    # `elas` is built directly in Python, so it has no `_hit_name` and gets
    # the `_child_<i>` fallback. `hard` came through the factory and carries
    # `_hit_name = "h"`, so it registers under the HIT block name.
    elas = LinearIsotropicElasticity(E=100.0, nu=0.3)
    composed = ComposedModel([elas, hard])
    names = {n for n, _ in composed.named_parameters()}
    assert {"_child_0.E", "_child_0.nu", "h.K"} <= names


# ---------------------------------------------------------------------------
# Autograd through a leaf
# ---------------------------------------------------------------------------


def test_autograd_through_leaf_matches_closed_form():
    """dloss/dE matches sum(σ²)/E since σ is linear in E.

    Note: ``model.E`` returns a typed ``Scalar(nn.Parameter(...))`` wrapper
    via :meth:`Model.__getattr__`; the underlying ``nn.Parameter`` (which
    carries ``.grad``) lives in ``model._parameters["E"]``.
    """
    model = LinearIsotropicElasticity(E=100.0, nu=0.3)
    strain = SR2(torch.tensor([0.09, 0.04, -0.02, 0.0, 0.0, 0.0]))
    stress = model(strain)
    assert isinstance(stress, SR2)
    loss = (stress.data**2).sum()
    loss.backward()

    params = dict(model.named_parameters())
    expected_dE = 2.0 * (stress.data.detach() ** 2).sum() / params["E"].detach()
    grad_E = params["E"].grad
    grad_nu = params["nu"].grad
    assert grad_E is not None and grad_nu is not None
    assert torch.allclose(grad_E, expected_dE, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(grad_nu)


def test_autograd_through_composed_chain_matches_finite_difference():
    """End-to-end autograd through a ComposedModel matches central-diff on E."""
    elas = LinearIsotropicElasticity(E=100.0, nu=0.3)
    composed = ComposedModel([elas])
    strain = torch.tensor([[0.09, 0.04, -0.02, 0.0, 0.0, 0.0]])

    # ComposedModel returns typed wrappers; unwrap to torch.Tensor for ** /
    # autograd backward.
    (stress,) = composed(strain)
    loss = (stress.data**2).sum()
    loss.backward()
    E_param = dict(elas.named_parameters())["E"]
    assert E_param.grad is not None
    analytical = E_param.grad.item()

    E0 = E_param.data.clone()
    h = 1e-3
    with torch.no_grad():
        E_param.data = E0 + h
        (stress_p,) = composed(strain)
        loss_p = (stress_p.data**2).sum().item()
        E_param.data = E0 - h
        (stress_m,) = composed(strain)
        loss_m = (stress_m.data**2).sum().item()
        E_param.data = E0
    fd = (loss_p - loss_m) / (2 * h)
    assert math.isclose(analytical, fd, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end calibration fit
# ---------------------------------------------------------------------------


def test_calibration_recovers_E_from_synthetic_stress():
    """Single-parameter optimisation through eager autograd: recover E.

    Generates synthetic stress under known E_true; calibrates the model's E
    starting from a perturbed initial guess. L-BFGS converges to within 0.1%
    of E_true in a handful of iterations.
    """
    E_true = 100.0
    nu_fixed = 0.3
    strains = torch.tensor(
        [
            [0.09, 0.04, -0.02, 0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.005, 0.005, 0.005],
        ]
    )
    with torch.no_grad():
        gold = LinearIsotropicElasticity(E_true, nu_fixed)
        targets = gold(SR2(strains)).data.clone()

    model = LinearIsotropicElasticity(E=70.0, nu=nu_fixed)
    params = dict(model.named_parameters())
    # Freeze nu so we calibrate E only.
    params["nu"].requires_grad_(False)

    optim = torch.optim.LBFGS([params["E"]], lr=1.0, max_iter=50, tolerance_grad=1e-12)

    def closure():
        optim.zero_grad()
        pred = model(SR2(strains)).data
        loss = ((pred - targets) ** 2).sum()
        loss.backward()
        return loss

    for _ in range(5):
        optim.step(closure)

    assert math.isclose(params["E"].item(), E_true, rel_tol=1e-3)
