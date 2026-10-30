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

"""Derivatives through, and with respect to, sub-batched variables.

Each case compiles a ``-d`` selection and checks the py-aoti blocks and forms
against the eager bridge, the native ``Model.jacobian`` and central differences.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import neml2  # noqa: F401 -- registers models

_SCN = Path(__file__).parent

# Same math on both sides, float64.
_RTOL_ROUTE, _ATOL_ROUTE = 1e-10, 1e-12
# Central differences of models whose residual is linear in the unknowns (one
# Newton step to round-off), h = 1e-6.
_H = 1e-6
_RTOL_FD, _ATOL_FD = 1e-6, 1e-8

# (scenario, -d specs, expected form of every compiled pair)
_CASES = [
    (
        "implicit_sub_batch_ift_reduce",
        ["vf:s", "vf:t", "x:s", "x:x~1", "vf:x~1", "y:z"],
        {
            ("vf", "s"): "plain",
            ("vf", "t"): "plain",
            ("x", "s"): "plain",
            ("x", "x~1"): "full",
            ("vf", "x~1"): "full",
            ("y", "z"): "paired",
        },
    ),
    (
        "implicit_sub_batch_ift_history",
        ["vf:s", "x:x~1", "vf:x~1", "y:z"],
        {
            ("vf", "s"): "plain",
            ("x", "x~1"): "full",
            ("vf", "x~1"): "full",
            ("y", "z"): "paired",
        },
    ),
    (
        # Not model.i: it declares a produced variable (g), which the generic
        # scenario sweep's input-only shape resolution rejects.
        "implicit_sub_batch_ift_given/given.i",
        [":", "x:x~1"],
        {
            ("vf", "s"): "plain",
            ("vf", "t"): "plain",
            ("vf", "t~1"): "plain",
            ("x", "s"): "plain",
            ("x", "t"): "plain",
            ("x", "t~1"): "plain",
            ("x", "x~1"): "full",
        },
    ),
    (
        "implicit_sub_batch_ift_coupled",
        ["vf:s", "x:s", "x:x~1", "vf:x~1"],
        {
            ("vf", "s"): "plain",
            ("x", "s"): "plain",
            ("x", "x~1"): "full",
            ("vf", "x~1"): "full",
        },
    ),
    (
        "implicit_grain_global_cross",
        [":", "u_per:g_per", "u_glob:g_per"],
        {
            ("u_per", "g_glob"): "plain",
            ("u_per", "u_glob"): "plain",
            ("u_per", "coupling"): "plain",
            ("u_glob", "g_glob"): "plain",
            ("u_glob", "u_glob"): "plain",
            ("u_glob", "coupling"): "plain",
            ("u_per", "g_per"): "full",
            ("u_glob", "g_per"): "full",
        },
    ),
    (
        "implicit_sub_batch_ift_2d",
        ["vf:s", "x:s", "x:x~1", "vf:x~1", "y:z"],
        {
            ("vf", "s"): "plain",
            ("x", "s"): "plain",
            ("x", "x~1"): "full",
            ("vf", "x~1"): "full",
            ("y", "z"): "paired",
        },
    ),
    ("forward_sub_batch", ["stress:strain"], {("stress", "strain"): "paired"}),
    ("forward_sub_batch_reduce", ["vf:x"], {("vf", "x"): "full"}),
    (
        "forward_sub_batch_plain_output",
        ["x_scaled:x", "y:t"],
        {("x_scaled", "x"): "paired", ("y", "t"): "plain"},
    ),
]


@pytest.fixture(autouse=True, scope="module")
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _inputs(eager, B: int = 3) -> dict[str, torch.Tensor]:
    """Random positive inputs at each input's declared ``(*B, *sub, *base)``."""
    g = torch.Generator().manual_seed(7)
    return {
        n: 0.1 + 0.5 * torch.rand(B, *sub, *base, generator=g, dtype=torch.float64)
        for n, sub, base in zip(
            eager.input_names, eager.input_sub_batch_shapes, eager.input_base_shapes, strict=True
        )
    }


def _fd_block(eager, raw, o: str, i: str, form: str, out_sub, in_sub) -> torch.Tensor:
    """Central-difference block of ``(o, i)`` in *form*; batch entries are
    independent, so each component is perturbed in all of them at once."""
    x = raw[i]
    B = x.shape[0]
    n_in = x[0].numel()
    base = eager.forward(raw)[o]
    cols = []
    for k in range(n_in):
        xp = x.reshape(B, n_in).clone()
        xm = xp.clone()
        xp[:, k] += _H
        xm[:, k] -= _H
        fp = eager.forward({**raw, i: xp.reshape(x.shape)})[o]
        fm = eager.forward({**raw, i: xm.reshape(x.shape)})[o]
        cols.append(((fp - fm) / (2 * _H)).reshape(B, -1))
    full = torch.stack(cols, dim=-1)  # (B, out_numel, in_numel)
    o_trail = tuple(base.shape[1:])
    i_trail = tuple(x.shape[1:])
    if form == "paired":
        S = math.prod(in_sub)
        grid = full.reshape(B, S, -1, S, n_in // S)
        diag = grid.diagonal(0, 1, 3).movedim(-1, 1)  # (B, S, out_base, in_base)
        o_base = o_trail[len(out_sub) :]
        i_base = i_trail[len(in_sub) :]
        return diag.reshape(B, *in_sub, *o_base, *i_base)
    return full.reshape(B, *o_trail, *i_trail)


def _hit(scenario: str) -> Path:
    return _SCN / scenario if scenario.endswith(".i") else _SCN / scenario / "model.i"


@pytest.mark.parametrize(
    ("scenario", "specs", "forms"), _CASES, ids=[c[0].split("/")[0] for c in _CASES]
)
def test_compiled_sub_batch_derivatives_match_eager_and_fd(tmp_path, scenario, specs, forms):
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel
    from neml2.types import Scalar  # noqa: F401 -- typed wrappers for the native call

    hit = _hit(scenario)
    meta = export_model_for_aoti(hit, "model", tmp_path / "art", derivatives=specs)
    assert {(o, i) for o, i in meta["derivatives"]} == set(forms)
    m = AOTIModel(str(tmp_path / "art"))
    assert {(o, i): f for o, row in m.derivative_forms.items() for i, f in row.items()} == forms

    eager = _EagerModel(str(hit), "model")
    raw = _inputs(eager)
    outc, Jc = m.jacobian(raw)
    oute, Je = eager.jacobian(raw, derivatives=specs)
    for o, v in oute.items():
        torch.testing.assert_close(outc[o], v, rtol=_RTOL_ROUTE, atol=_ATOL_ROUTE, msg=o)

    native = neml2.load_model(str(hit), "model")
    typed = {
        n: native.input_spec[n](t, sub_batch_ndim=len(sub))
        for (n, t), sub in zip(raw.items(), eager.input_sub_batch_shapes, strict=True)
    }
    _, Jn = native.jacobian(typed, derivatives=specs)

    out_sub = dict(zip(eager.output_names, eager.output_sub_batch_shapes, strict=True))
    in_sub = dict(zip(eager.input_names, eager.input_sub_batch_shapes, strict=True))
    for (o, i), form in forms.items():
        # A batch-independent compiled block comes back unbatched; it must
        # broadcast against the batched block of the other routes.
        fd = _fd_block(eager, raw, o, i, form, out_sub[o], in_sub[i])
        c, e, n, fd = torch.broadcast_tensors(Jc[o][i], Je[o][i], Jn[o][i].data, fd)
        assert Je[o][i].shape == fd.shape, f"{o}:{i} eager block shape"
        torch.testing.assert_close(c, e, rtol=_RTOL_ROUTE, atol=_ATOL_ROUTE, msg=f"{o}:{i} aoti")
        torch.testing.assert_close(n, e, rtol=_RTOL_ROUTE, atol=_ATOL_ROUTE, msg=f"{o}:{i} native")
        torch.testing.assert_close(
            e, fd, rtol=_RTOL_FD, atol=_ATOL_FD, msg=f"{o}:{i} finite differences"
        )


def test_reduction_through_solve_matches_closed_form(tmp_path):
    """``d vf / d s = sum_i 4/3 pi R_i^3 dt / (1 - k_i dt)`` with per-site rates k_i."""
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    hit = _SCN / "implicit_sub_batch_ift_reduce" / "model.i"
    export_model_for_aoti(hit, "model", tmp_path / "art", derivatives=["vf:s"])
    m = AOTIModel(str(tmp_path / "art"))
    B, dt = 3, 0.1
    g = torch.Generator().manual_seed(3)
    raw = {
        "x": torch.rand(B, 4, generator=g),
        "x~1": torch.rand(B, 4, generator=g),
        "s": torch.rand(B, generator=g),
        "t": torch.full((B,), dt),
        "t~1": torch.zeros(B),
        "z": torch.rand(B, 4, generator=g),
    }
    _, J = m.jacobian(raw)
    R = torch.logspace(-1.0, 0.0, 4)
    k = -torch.logspace(0.0, 1.0, 4)
    exact = (4.0 / 3.0 * math.pi * R**3 * dt / (1.0 - k * dt)).sum()
    torch.testing.assert_close(J["vf"]["s"], exact.expand(B), rtol=1e-12, atol=1e-14)


def test_wildcard_selects_plain_inputs_and_names_select_sub_batched(tmp_path):
    from neml2.cli.aoti_export import export_model_for_aoti

    hit = _SCN / "implicit_sub_batch_ift_reduce" / "model.i"
    meta = export_model_for_aoti(hit, "model", tmp_path / "a", derivatives=[":"])
    ins = {i for _, i in meta["derivatives"]}
    assert ins == {"s", "t", "t~1"}, "a wildcard must skip the sub-batched x, x~1 and z"
    assert set(meta["derivative_forms"]) == {"plain"}

    meta = export_model_for_aoti(hit, "model", tmp_path / "b", derivatives=["y:", "y:z"])
    assert [tuple(p) for p in meta["derivatives"]] == [
        ("y", "s"),
        ("y", "t"),
        ("y", "t~1"),
        ("y", "z"),
    ]
    assert meta["derivative_forms"] == ["plain", "plain", "plain", "paired"]


def test_eager_selection_matches_compile_selection():
    from neml2.eager import _EagerModel

    eager = _EagerModel(str(_SCN / "implicit_sub_batch_ift_reduce" / "model.i"), "model")
    raw = _inputs(eager)
    _, J = eager.jacobian(raw, derivatives=[":"])
    assert {(o, i) for o, row in J.items() for i in row} == {
        (o, i) for o in ("y", "vf", "x") for i in ("s", "t", "t~1")
    }
    _, J = eager.jacobian(raw, derivatives=["vf:x~1"])
    assert list(J) == ["vf"] and list(J["vf"]) == ["x~1"]
    assert J["vf"]["x~1"].shape == (3, 4)


def test_aoti_jvp_with_sub_batched_tangent(tmp_path):
    from neml2.aoti import Model as AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti
    from neml2.eager import _EagerModel

    hit = _SCN / "implicit_sub_batch_ift_reduce" / "model.i"
    export_model_for_aoti(hit, "model", tmp_path / "art", derivatives=["vf:s", "vf:x~1"])
    m = AOTIModel(str(tmp_path / "art"))
    raw = _inputs(_EagerModel(str(hit), "model"))
    _, J = m.jacobian(raw)
    t = torch.rand(3)
    tz = torch.rand(3, 4)
    _, jv = m.jvp(raw, {"s": t, "x~1": tz})
    expect = J["vf"]["s"] * t + (J["vf"]["x~1"] * tz).sum(-1)
    torch.testing.assert_close(jv["vf"], expect, rtol=_RTOL_ROUTE, atol=_ATOL_ROUTE)


_BLOCK_I = """
[Settings]
  [example_batch_shape]
    u = '(2; 3)'
    g = '(2; 3)'
  []
[]
[Models]
  [residual]
    type = ScalarLinearCombination
    from = 'u g'
    to = 'r'
    weights = '2 -1'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]
[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'residual'
    unknowns = 'u'
    residuals = 'r'
    structure = 'block'
  []
[]
[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]
"""


def test_block_structured_sub_batched_solve_rejected_at_compile(tmp_path):
    """BLOCK groups are rejected, as on eager."""
    from neml2.cli.aoti_export import export_model_for_aoti

    hit = tmp_path / "block.i"
    hit.write_text(_BLOCK_I)
    with pytest.raises(NotImplementedError, match="sub-batched.*BLOCK-structured"):
        export_model_for_aoti(hit, "model", tmp_path / "a", derivatives=["u:g"])


def test_iterative_input_sensitivity_with_sub_batched_pair_rejected(tmp_path):
    from neml2.cli.aoti_export import export_model_for_aoti

    src = (_SCN / "implicit_sub_batch_ift_reduce" / "model.i").read_text()
    src = src.replace(
        "    solver = 'newton'\n  []\n  [vf]",
        "    solver = 'newton'\n    input_sensitivity_solver = 'gmres'\n  []\n  [vf]",
    ).replace("  [lu]\n", "  [gmres]\n    type = GMRES\n  []\n  [lu]\n")
    hit = tmp_path / "gmres.i"
    hit.write_text(src)
    with pytest.raises(NotImplementedError, match="iterative input-sensitivity"):
        export_model_for_aoti(hit, "model", tmp_path / "a", derivatives=["vf:s"])


def test_parameter_derivative_through_sub_batched_solve_rejected(tmp_path):
    from neml2.cli.aoti_export import export_model_for_aoti

    hit = _SCN / "implicit_sub_batch_ift_given" / "given.i"
    q = "update.residual.rate.weight_1"
    with pytest.raises(NotImplementedError, match="parameter derivatives.*sub-batched"):
        export_model_for_aoti(hit, "model", tmp_path / "a", promoted={q}, derivatives=[f"vf:{q}"])
