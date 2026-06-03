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

"""
Round-trip: Python eager nn.Module → torch.export → AOTI compile → AOTI load
in Python → numerically equivalent to eager.

AOTI packaging in this test covers the pure-forward (v=None) path.  The
Jacobian pushforward path (v=...) is covered by the ComposedModel export tests.
"""

import pytest
import torch

from neml2.export import compile_model, load_package
from neml2.models.solid_mechanics.elasticity import LinearIsotropicElasticity
from neml2.types import SR2

E = 100.0
NU = 0.3


def _example_strain(batch: int = 2, dtype: torch.dtype = torch.float64) -> SR2:
    return SR2(torch.randn(batch, 6, dtype=dtype))


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.zeros(1, device="cuda") + 1
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


@pytest.mark.aoti_compile
def test_export_round_trip_matches_eager_cpu(tmp_path):
    model = LinearIsotropicElasticity(E, NU)
    package = compile_model(model, (_example_strain(batch=2),), tmp_path / "elasticity.pt2")
    assert package.exists()

    loaded = load_package(package)

    for batch in (1, 8, 256):
        strain = _example_strain(batch=batch)
        out_eager = model(strain)
        out_loaded = loaded(strain)
        assert isinstance(out_loaded, SR2)
        assert torch.allclose(out_loaded.data, out_eager.data, rtol=1e-12, atol=1e-12)


@pytest.mark.aoti_compile
@pytest.mark.skipif(not _cuda_usable(), reason="No usable CUDA device for this torch build")
def test_export_round_trip_matches_eager_cuda(tmp_path):
    model = LinearIsotropicElasticity(E, NU).cuda()
    example = (_example_strain(batch=2).to("cuda"),)
    package = compile_model(model, example, tmp_path / "elasticity_cuda.pt2")

    loaded = load_package(package)

    for batch in (1, 8, 256):
        strain = _example_strain(batch=batch).to("cuda")
        out_eager = model(strain)
        out_loaded = loaded(strain)
        assert isinstance(out_loaded, SR2)
        assert torch.allclose(out_loaded.data, out_eager.data, rtol=1e-12, atol=1e-12)
