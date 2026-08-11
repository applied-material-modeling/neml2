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

"""The predictor runtime loop, through the compiled route.

A predictor that is a bounded iteration -- coordinate descent here -- is not
unrolled into the exported graph. The exporter compiles a *single* step taking
the previous step's value on an extra input, and the C++ runtime re-runs the
graph, feeding the output back.

Checking the converged answer is necessary but nowhere near sufficient: a
predictor cannot change the root, so the compiled model would agree with eager
even if the loop ran once or not at all. The test that actually pins the loop is
:func:`test_the_runtime_really_iterates`, which lowers the iteration count in the
metadata and shows the Newton work go up.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_MODEL = _REPO / "tests" / "aoti" / "implicit_predictor_loop" / "model.i"
_NAME = "model_with_stress"
_NBATCH = 2
_NSLIP = 12
#: Matches the CoordinateDescentPredictor `sweeps` in the fixture.
_SWEEPS = 16


def _implicit_seg(meta: dict) -> dict:
    (seg,) = [s for s in meta["segments"] if s["kind"] == "implicit"]
    return seg


def _inputs() -> dict[str, torch.Tensor]:
    b = _NBATCH
    return {
        "deformation_rate": torch.tensor([0.1, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=torch.float64)
        .expand(b, 6)
        .contiguous(),
        "vorticity": torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64)
        .expand(b, 3)
        .contiguous(),
        "t": torch.full((b,), 0.0101, dtype=torch.float64),
        "t~1": torch.zeros(b, dtype=torch.float64),
        "elastic_strain~1": torch.zeros(b, 6, dtype=torch.float64),
        "slip_hardening~1": torch.zeros(b, dtype=torch.float64),
        "orientation~1": torch.zeros(b, 3, dtype=torch.float64),
    }


@pytest.fixture(scope="module")
def compiled(tmp_path_factory) -> Path:
    """Compile once; the Inductor pass dominates this module's runtime."""
    from neml2.cli.aoti_export import export_model_for_aoti

    out = tmp_path_factory.mktemp("predictor_loop") / _NAME
    export_model_for_aoti(_MODEL, _NAME, out)
    return out


def test_metadata_carries_the_feedback_pair(compiled: Path):
    """The runtime cannot loop without knowing what to feed back, how often, and
    at what shape to seed the first pass."""
    meta = json.loads((compiled / "metadata.json").read_text())
    fb = _implicit_seg(meta)["predictor_feedback"]

    assert fb["iterations"] == _SWEEPS
    assert fb["output"] == "slip_rates"
    assert fb["name"] == "slip_rates_in"
    # The seed shape is measured at export, not guessed: the slip-system count
    # reaches the predictor only through the coupling matrix.
    assert fb["sub_batch_shape"] == [_NSLIP]
    assert fb["base_shape"] == []

    seg = _implicit_seg(meta)
    names_in = [v["name"] for v in seg["predictor_inputs"]]
    names_out = [v["name"] for v in seg["predictor_outputs"]]
    assert fb["name"] in names_in, "feedback input must be a graph input"
    assert fb["output"] in names_out, "feedback output must be a graph output"


def test_compiled_matches_eager(compiled: Path):
    """py-aoti (C++ loop) against py-eager (Python loop) on the same model."""
    from neml2 import load_input
    from neml2.aoti import Model as AOTIModel

    aoti = AOTIModel(str(compiled))
    eager = load_input(_MODEL).get_model(_NAME)

    raw = _inputs()
    got = aoti.forward({k: raw[k] for k in aoti.input_names})
    typed = tuple(eager.input_spec[k](raw[k]) for k in eager.input_spec)
    ref_out = eager(*typed)
    ref = dict(zip(eager.output_spec, ref_out, strict=True))

    for name, want in ref.items():
        assert torch.allclose(got[name], want.data.detach(), rtol=1e-10, atol=1e-12), (
            f"{name} differs between the compiled and eager routes"
        )


def test_the_runtime_really_iterates(compiled: Path, tmp_path: Path):
    """Lower the loop count and the Newton solve has to work harder.

    This is the test that distinguishes a working loop from a loop that never
    runs. Agreement on the converged answer cannot: a predictor only moves the
    starting point, so every iteration count converges to the same root. What
    changes is how far Newton has to travel, and the solver reports that.

    Also demonstrates that ``iterations`` is honoured from metadata -- it is
    retunable without recompiling, which is the point of not baking it into the
    graph.
    """
    import neml2.log as nlog
    from neml2.aoti import Model as AOTIModel

    def newton_iters(root: Path) -> int:
        # Capture through the log sink rather than the NEML2_LOGS env var: the
        # env layer is read once per process, so setting it from inside a test
        # that has already loaded the runtime does nothing.
        lines: list[str] = []
        nlog.set_default_level("newton", "info")
        nlog.set_sink(lambda _level, line: lines.append(line))
        try:
            model = AOTIModel(str(root))
            raw = _inputs()
            model.forward({k: raw[k] for k in model.input_names})
        finally:
            nlog.reset_sink()
            nlog.reset_defaults()
        found = re.findall(r"converged \(iters=(\d+)", "\n".join(lines))
        assert found, (
            "no Newton convergence line captured; the log format or channel name "
            f"may have changed. lines={lines!r}"
        )
        return int(found[0])

    # A copy with the loop cut to a single sweep. Only the metadata changes --
    # the compiled graph is byte-identical, which is what makes this a clean
    # measurement of the loop rather than of two different programs.
    one = tmp_path / "one_sweep"
    shutil.copytree(compiled, one)
    meta_path = one / "metadata.json"
    meta = json.loads(meta_path.read_text())
    _implicit_seg(meta)["predictor_feedback"]["iterations"] = 1
    meta_path.write_text(json.dumps(meta))

    full_sweeps = newton_iters(compiled)
    single_sweep = newton_iters(one)
    assert single_sweep > full_sweeps, (
        f"cutting the predictor loop from {_SWEEPS} sweeps to 1 left the Newton "
        f"count at {single_sweep} vs {full_sweeps} -- the runtime loop is not running"
    )
