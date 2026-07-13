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

"""AOTI compiled matrix-free Krylov path: metadata contract + correctness.

The generic ``tests/aoti/test_aoti.py`` sweep already checks that the compiled
Krylov artifacts (``krylov_gmres`` / ``krylov_gmres_precond``) reproduce the eager
solve (py-eager == py-aoti/cpp-aoti). This file pins the two things that sweep
does not: (1) the schema-v11 ``solver_config`` metadata contract (``solver_kind``
+ nested ``krylov`` block; ``matvec`` graph in place of ``solve``); and (2) that
the iterative result is *correct*, i.e. equals the direct (DenseLU) solve of the
same model on the same inputs -- not merely self-consistent between routes.
"""

from __future__ import annotations

from pathlib import Path

import torch

_HERE = Path(__file__).parent
_GMRES = _HERE / "krylov_gmres" / "model.i"
_GMRES_PC = _HERE / "krylov_gmres_precond" / "model.i"
# The same physics with the default direct DenseLU solve (the correctness gold).
_DIRECT = _HERE.parent / "regression/solid_mechanics/viscoplasticity/perfect/model.i"


def _inputs(input_spec: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """A reproducible plain-batch (dyn=(4,)) raw-tensor input dict."""
    gen = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(4, *tuple(type_cls.BASE_SHAPE), generator=gen, dtype=torch.float64)
        for name, type_cls in input_spec.items()
    }


def test_krylov_metadata_contract(tmp_path: Path):
    """A matrix-free GMRES export records solver_kind=krylov + the krylov block,
    and emits the matvec graph in place of the baked solve (no preconditioner ->
    no jacobian either)."""
    from neml2.cli.aoti_export import AOTI_META_SCHEMA_VERSION, export_model_for_aoti

    out = tmp_path / "gmres"
    meta = export_model_for_aoti(_GMRES, "model", out)
    assert meta["schema_version"] == AOTI_META_SCHEMA_VERSION

    sc = meta["solver_config"]
    assert sc["solver_kind"] == "krylov"
    assert sc["krylov"]["method"] == "gmres"
    assert sc["krylov"]["preconditioner"] == "none"
    assert sc["krylov"]["cache_strategy"] == "none"

    seg = next(s for s in meta["segments"] if s.get("kind") == "implicit")
    # Matrix-free, no preconditioner: residual + matvec only.
    assert "matvec_package" in seg
    assert "solve_package" not in seg
    assert "jacobian_package" not in seg


def test_krylov_precond_metadata_contract(tmp_path: Path):
    """A preconditioned GMRES export additionally emits the jacobian graph (the
    C++ builds the preconditioner from the assembled A)."""
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "gmres_pc"
    meta = export_model_for_aoti(_GMRES_PC, "model", out)
    sc = meta["solver_config"]
    assert sc["solver_kind"] == "krylov"
    assert sc["krylov"]["preconditioner"] == "block_jacobi"
    assert sc["krylov"]["cache_strategy"] == "chord"

    seg = next(s for s in meta["segments"] if s.get("kind") == "implicit")
    assert "matvec_package" in seg
    assert "jacobian_package" in seg  # needed to assemble A for the preconditioner
    assert "solve_package" not in seg


def test_krylov_aoti_matches_direct_solve(tmp_path: Path):
    """The compiled matrix-free GMRES solve lands on the same converged stress as
    the direct DenseLU solve of the identical model -- correctness, not just
    route-to-route self-consistency."""
    from neml2 import load_input
    from neml2.aoti import AOTIModel
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path / "gmres"
    export_model_for_aoti(_GMRES, "model", out)
    aoti = AOTIModel(str(out))

    direct = load_input(_DIRECT).get_model("model")
    raw = _inputs(dict(direct.input_spec))

    # The shim's public ``forward`` is typed-positional (it plays the native Model
    # role); the raw-dict forward is the binding handle's interface.
    aoti_out = aoti._inner.forward(raw)
    direct_args = tuple(direct.input_spec[n](raw[n]) for n in direct.input_spec)
    direct_raw = direct(*direct_args)
    direct_tuple = direct_raw if isinstance(direct_raw, tuple) else (direct_raw,)
    direct_out = {
        name: (v.data if hasattr(v, "data") else v)
        for name, v in zip(direct.output_spec, direct_tuple, strict=True)
    }

    for name, gold in direct_out.items():
        got = aoti_out[name]
        assert torch.allclose(got, gold, rtol=1e-8, atol=1e-8), (
            f"krylov-AOTI vs direct mismatch for {name!r} "
            f"(max abs diff {(got - gold).abs().max().item():.3e})"
        )
