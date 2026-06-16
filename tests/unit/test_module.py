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

import pytest
import torch

from neml2.models.model import Model
from neml2.types import SR2, Scalar

# NEML2Module is an alias for Model, retained for test readability.
NEML2Module = Model


class _ToyModel(Model):
    input_spec = {}
    output_spec = {}

    def __init__(self, E: float, sigma_init: torch.Tensor):
        super().__init__()
        self.register_typed_buffer("E", Scalar(torch.as_tensor(E)))
        self.register_typed_buffer("sigma0", SR2(sigma_init))

    def forward(self):  # type: ignore[override]
        pass


def test_typed_buffer_returns_wrapper_on_read():
    m = _ToyModel(100.0, torch.arange(6.0))
    assert isinstance(m.E, Scalar)
    assert m.E.data.item() == 100.0
    assert isinstance(m.sigma0, SR2)
    assert torch.equal(m.sigma0.data, torch.arange(6.0))


def test_state_dict_has_raw_tensors():
    m = _ToyModel(100.0, torch.arange(6.0))
    sd = m.state_dict()
    assert set(sd.keys()) == {"E", "sigma0"}
    for v in sd.values():
        assert isinstance(v, torch.Tensor)
        assert not hasattr(v, "BASE_NDIM"), "state_dict must hold raw tensors, not typed wrappers"


def test_load_state_dict_round_trip():
    src = _ToyModel(100.0, torch.arange(6.0))
    dst = _ToyModel(0.0, torch.zeros(6))
    dst.load_state_dict(src.state_dict())
    assert dst.E.data.item() == 100.0  # type: ignore[reportCallIssue]
    assert torch.equal(dst.sigma0.data, torch.arange(6.0))  # type: ignore[reportArgumentType]


def test_to_dtype_survives():
    m = _ToyModel(100.0, torch.arange(6.0, dtype=torch.float64))
    m32 = m.to(torch.float32)
    assert m32.E.dtype == torch.float32
    assert m32.sigma0.dtype == torch.float32


def test_non_typed_attributes_pass_through():
    """`__getattr__` shouldn't break submodule / non-buffer attribute access."""

    class WithChild(Model):
        input_spec = {}
        output_spec = {}

        def __init__(self):
            super().__init__()
            self.register_typed_buffer("E", Scalar(torch.as_tensor(1.0)))
            self.child = torch.nn.Linear(3, 3)

        def forward(self):  # type: ignore[override]
            pass

    m = WithChild()
    assert isinstance(m.child, torch.nn.Linear)
    assert isinstance(m.E, Scalar)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_to_cuda_survives():
    m = _ToyModel(100.0, torch.arange(6.0))
    m_cuda = m.to("cuda")
    assert m_cuda.E.device.type == "cuda"
    assert m_cuda.sigma0.device.type == "cuda"
    # And the wrapper-on-read still returns the correct types
    assert isinstance(m_cuda.E, Scalar)
    assert isinstance(m_cuda.sigma0, SR2)
