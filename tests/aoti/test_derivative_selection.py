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

"""End-to-end tests for selectable derivative pairs (``neml2-compile -d``).

These compile real artifacts and drive the public
:class:`~neml2.aoti.AOTIModel` shim's ``jvp`` / ``jacobian`` (which delegate to
``neml2.aoti.Model``, the C++ binding), checking the runtime contract against
the eager chain rule
(:class:`neml2.eager._EagerModel`): default-off raises, a selected pair returns
only that pair and matches eager, batch-independent blocks come back unbatched,
and the composed-model reachability prune stays numerically correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import neml2  # noqa: F401 — registers models

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_ELASTICITY_I = _TESTS_ROOT / "models/solid_mechanics/elasticity/LinearIsotropicElasticity.i"
_COMPOSED_I = _TESTS_ROOT / "aoti/forward_implicit/model.i"
# Two independent pipelines (forward strain->mandel_stress; implicit x->y) in one
# model: requesting one pipeline's pair prunes the OTHER pipeline's whole segment.
_PRUNE_I = _TESTS_ROOT / "aoti/derivative_prune/prune.i"
# A sub-batched (per-grain) implicit model: unknowns u_per (per-grain) / u_glob
# (global); givens g_per / g_glob / coupling.
_SUBBATCH_IMPLICIT_I = _TESTS_ROOT / "aoti/implicit_grain_global_cross/model.i"


def _broadcast_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if *a* equals *b* after broadcasting (a batch-independent block is
    returned unbatched, so it must broadcast-match the eager batched block)."""
    a = a.expand_as(b) if a.shape != b.shape else a
    return torch.allclose(a, b, rtol=1e-6, atol=1e-8)


@pytest.fixture(scope="module", autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _compile(tmp_path: Path, hit: Path, name: str, derivatives):
    from neml2.aoti import AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "art"
    export_model_for_aoti(hit, name, out, derivatives=derivatives)
    # Drive the public AOTIModel shim; its jvp/jacobian delegate to the binding.
    # Schema v10: pass the artifact ROOT (metadata.json + <device>/<dtype>/).
    return AOTIModel(str(out))


def _rand_inputs(eager, B=5):
    g = torch.Generator().manual_seed(0)
    return {
        n: torch.randn(B, *eager.input_spec[n].BASE_SHAPE, generator=g, dtype=torch.float64) * 1e-3
        for n in eager.input_names
    }


def test_default_off_jvp_jacobian_raise(tmp_path):
    """No ``-d`` flags -> no derivative graphs -> jvp/jacobian raise, pointing
    the user at ``-d``."""
    m = _compile(tmp_path, _ELASTICITY_I, "model", derivatives=())
    strain = torch.randn(5, 6, dtype=torch.float64) * 1e-2
    with pytest.raises(Exception, match="-d"):
        m.jacobian({"strain": strain})
    with pytest.raises(Exception, match="-d"):
        m.jvp({"strain": strain}, {"strain": strain})


def test_selected_pair_returns_only_that_pair_and_matches_eager(tmp_path):
    from neml2.eager import _EagerModel

    m = _compile(tmp_path, _ELASTICITY_I, "model", derivatives=["stress:strain"])
    eager = _EagerModel(str(_ELASTICITY_I), "model")
    ins = _rand_inputs(eager)

    _, Jc = m.jacobian(ins)
    _, Je = eager.jacobian(ins)
    assert list(Jc) == ["stress"] and list(Jc["stress"]) == ["strain"]
    assert _broadcast_equal(Jc["stress"]["strain"], Je["stress"]["strain"])

    _, jv = m.jvp(ins, ins)
    _, jve = eager.jvp(ins, ins)
    assert list(jv) == ["stress"]
    assert torch.allclose(jv["stress"], jve["stress"], rtol=1e-6, atol=1e-8)


def test_jvp_missing_tangent_contributes_zero(tmp_path):
    """A tangent dict missing an input key contributes nothing: the runtime
    packs an absent tangent as zeros, so jvp along an empty tangent is zero."""
    from neml2.eager import _EagerModel

    m = _compile(tmp_path, _ELASTICITY_I, "model", derivatives=["stress:strain"])
    eager = _EagerModel(str(_ELASTICITY_I), "model")
    ins = _rand_inputs(eager)

    _, jv = m.jvp(ins, {})  # no tangents -> zero directional derivative
    assert torch.allclose(jv["stress"], torch.zeros_like(jv["stress"]))


def test_constant_jacobian_block_returned_unbatched(tmp_path):
    """Linear elasticity's d(stress)/d(strain) is the constant stiffness tensor:
    the metadata flags it ``batch_independent`` and the single-forward-segment
    Jacobian path returns it **unbatched** at its natural ``(6, 6)`` (no leading
    batch axis), regardless of the runtime batch size."""
    import json

    from neml2.aoti import AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "art"
    meta = export_model_for_aoti(_ELASTICITY_I, "model", out, derivatives=["stress:strain"])
    on_disk = json.loads((out / "metadata.json").read_text())
    assert meta["derivatives"] == [["stress", "strain"]]
    pair = on_disk["segments"][0]["jacobian_pairs"][0]
    assert (pair["out_var"], pair["in_var"]) == ("stress", "strain")
    assert pair["batch_independent"] is True

    m = AOTIModel(str(out))
    for B in (1, 7):
        _, J = m.jacobian({"strain": torch.randn(B, 6, dtype=torch.float64) * 1e-2})
        # Unbatched: natural (6, 6), no leading batch axis, independent of B.
        assert tuple(J["stress"]["strain"].shape) == (6, 6)


def test_composed_reachability_partial_matches_eager(tmp_path):
    """A composed forward+implicit model with a single requested master pair:
    the reachability prune drops off-path local pairs but the returned block is
    still numerically identical to the eager all-pairs result."""
    from neml2.eager import _EagerModel

    eager = _EagerModel(str(_COMPOSED_I), "model")
    ins = _rand_inputs(eager, B=4)
    _, Je = eager.jacobian(ins)
    o, i = next((o, i) for o in Je for i in Je[o])

    m = _compile(tmp_path, _COMPOSED_I, "model", derivatives=[f"{o}:{i}"])
    _, Jc = m.jacobian(ins)
    assert list(Jc) == [o] and list(Jc[o]) == [i]
    assert _broadcast_equal(Jc[o][i], Je[o][i])


@pytest.mark.parametrize(
    "pair",
    [
        # Forward-pipeline pair: keeps the forward segment, prunes the implicit one.
        "mandel_stress:strain",
        # Implicit-pipeline pair: keeps the implicit segment, prunes the forward one(s).
        "y:x",
    ],
)
def test_offpath_segment_pruned_jacobian_matches_eager(tmp_path, pair):
    """A composed model of two INDEPENDENT pipelines: requesting one pipeline's
    pair prunes the other pipeline's whole segment (no jvp/ift graph). At runtime
    that segment is advanced value-only and its dstate zero-filled; the requested
    block must still equal the eager chain-rule block (the off-path zeros are
    never folded into a kept pair)."""
    from neml2.eager import _EagerModel

    out, inp = (s.strip() for s in pair.split(":"))
    m = _compile(tmp_path, _PRUNE_I, "model", derivatives=[pair])
    eager = _EagerModel(str(_PRUNE_I), "model")
    ins = _rand_inputs(eager, B=4)

    _, Jc = m.jacobian(ins)
    _, Je = eager.jacobian(ins)
    assert list(Jc) == [out] and list(Jc[out]) == [inp]
    assert _broadcast_equal(Jc[out][inp], Je[out][inp])


def test_subbatch_implicit_global_global_compiles_and_matches_fd(tmp_path):
    """A sub-batched (crystal-plasticity-style) implicit model supports compiled
    Jacobian for a **non-sub-batched output w.r.t. non-sub-batched input**
    (``u_glob:g_glob``): the IFT solve handles the internal per-grain coupling and
    the returned global-to-global block has no grain axis. Validated by finite
    differences on the (sub-batch-capable) compiled forward."""
    from neml2.aoti import Model
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "art"
    meta = export_model_for_aoti(_SUBBATCH_IMPLICIT_I, "model", out, derivatives=["u_glob:g_glob"])
    assert meta["derivatives"] == [["u_glob", "g_glob"]]
    info = {v["name"]: v for v in meta["inputs"]}
    m = Model(str(out))

    B = 4
    g = torch.Generator().manual_seed(0)
    ins = {
        n: torch.randn(
            B,
            *info[n].get("sub_batch_shape", []),
            *info[n]["base_shape"],
            generator=g,
            dtype=torch.float64,
        )
        * 0.1
        + 0.5
        for n in m.input_names
    }
    _, J = m.jacobian(ins)
    block = J["u_glob"]["g_glob"]

    # Finite-difference reference on the compiled forward.
    gsz = max(1, int(torch.tensor(info["g_glob"]["base_shape"]).prod()))
    eps = 1e-6
    out0 = m.forward(ins)["u_glob"]
    cols = []
    for k in range(gsz):
        bumped = {n: t.clone() for n, t in ins.items()}
        flat = bumped["g_glob"].reshape(B, -1)
        flat[:, k] += eps
        bumped["g_glob"] = flat.reshape(ins["g_glob"].shape)
        cols.append(((m.forward(bumped)["u_glob"] - out0) / eps).reshape(B, -1))
    fd = torch.stack(cols, dim=-1)
    assert torch.allclose(block.reshape(fd.shape), fd, rtol=1e-3, atol=1e-5)


def test_subbatch_implicit_per_grain_pair_errors_at_compile(tmp_path):
    """Requesting a derivative that touches a sub-batched (per-grain) implicit
    variable fails fast at ``neml2-compile`` with a clear message (the per-pair
    IFT consumer is plain-batch only)."""
    from neml2.cli.aoti_export import export_model_for_aoti

    with pytest.raises(NotImplementedError, match="sub-batched"):
        export_model_for_aoti(_SUBBATCH_IMPLICIT_I, "model", tmp_path / "a", derivatives=[":"])


def test_all_pairs_parity_with_eager_composed(tmp_path):
    """``-d :`` on a composed model reproduces the full eager Jacobian."""
    from neml2.eager import _EagerModel

    eager = _EagerModel(str(_COMPOSED_I), "model")
    ins = _rand_inputs(eager, B=4)
    _, Je = eager.jacobian(ins)

    m = _compile(tmp_path, _COMPOSED_I, "model", derivatives=[":"])
    _, Jc = m.jacobian(ins)
    for o in Je:
        for i in Je[o]:
            assert i in Jc.get(o, {}), f"missing pair ({o}, {i})"
            assert _broadcast_equal(Jc[o][i], Je[o][i]), f"mismatch ({o}, {i})"
