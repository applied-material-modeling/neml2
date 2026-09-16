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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from neml2._accelerator import (
    KNOWN_FAMILIES,
    TOOLCHAIN_CHECKS,
    check_toolchain,
    folder_name,
    parse_device_spec,
    target_accelerator_family,
)


class TestParseDeviceSpec:
    @pytest.mark.parametrize(
        "spec, expected_type, expected_index",
        [
            ("cpu", "cpu", None),
            ("cuda", "cuda", None),
            ("cuda:0", "cuda", 0),
            ("cuda:1", "cuda", 1),
            ("xpu", "xpu", None),
            ("xpu:0", "xpu", 0),
            ("hip", "hip", None),
            ("hip:0", "hip", 0),
            ("mps", "mps", None),
        ],
    )
    def test_accepts_known(self, spec, expected_type, expected_index):
        dev = parse_device_spec(spec)
        assert dev.type == expected_type
        assert dev.index == expected_index

    def test_rejects_unknown_family(self):
        # torch itself refuses the string; we rewrap as ValueError with our own
        # allowed-set message rather than surface the torch-specific listing.
        with pytest.raises(ValueError, match="unknown device family"):
            parse_device_spec("foo")

    def test_rejects_unknown_with_index(self):
        with pytest.raises(ValueError, match="unknown device family"):
            parse_device_spec("bar:0")

    def test_rejects_known_to_torch_but_not_neml2(self):
        # ``meta`` parses fine in torch but is not in KNOWN_FAMILIES today; make
        # sure it's rejected with the neml2-family message rather than sneaking
        # through as a valid target.
        with pytest.raises(ValueError, match="unknown device family 'meta'"):
            parse_device_spec("meta")


class TestFolderName:
    @pytest.mark.parametrize("fam", KNOWN_FAMILIES)
    def test_round_trip_family_string(self, fam):
        assert folder_name(fam) == fam

    def test_round_trip_torch_device(self):
        for fam in KNOWN_FAMILIES:
            assert folder_name(torch.device(fam)) == fam

    def test_index_stripped(self):
        assert folder_name(torch.device("cuda", 1)) == "cuda"
        assert folder_name(torch.device("xpu", 0)) == "xpu"


class TestTargetAcceleratorFamily:
    def test_empty(self):
        assert target_accelerator_family(iter(())) is None

    def test_all_cpu(self):
        ts = [torch.zeros(2), torch.ones(3)]
        assert target_accelerator_family(iter(ts)) is None

    def test_meta_accelerator_is_returned(self):
        # A "meta" tensor is CPU-classified by torch (device.type == "meta"),
        # but the function is about non-cpu families, so meta counts.
        t = torch.zeros(2, device="meta")
        assert target_accelerator_family(iter([t])) == "meta"

    def test_first_nonzero_wins(self):
        # Fake tensor-like with .device.type via SimpleNamespace, so the test
        # runs without any real accelerator. The function's ordering contract
        # is what we're testing; it only reads .device.type.
        from typing import cast

        cpu = cast(torch.Tensor, SimpleNamespace(device=torch.device("cpu")))
        xpu = cast(torch.Tensor, SimpleNamespace(device=torch.device("xpu")))
        cuda = cast(torch.Tensor, SimpleNamespace(device=torch.device("cuda")))
        assert target_accelerator_family(iter([cpu, xpu, cuda])) == "xpu"


class TestCheckToolchain:
    def test_cpu_is_noop(self):
        check_toolchain("cpu")  # must not raise

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="no toolchain preflight registered"):
            check_toolchain("nvptx")

    def test_all_known_families_have_a_check(self):
        assert set(TOOLCHAIN_CHECKS) == set(KNOWN_FAMILIES)

    def test_cuda_ok_when_nvcc_on_path(self):
        with patch("shutil.which", return_value="/usr/local/cuda/bin/nvcc"):
            check_toolchain("cuda")  # must not raise

    def test_cuda_ok_when_nvcc_under_cuda_home(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "nvcc").touch()
        with (
            patch("shutil.which", return_value=None),
            patch("torch.utils.cpp_extension.CUDA_HOME", str(tmp_path)),
        ):
            check_toolchain("cuda")  # must not raise

    def test_cuda_raises_when_missing(self):
        with (
            patch("shutil.which", return_value=None),
            patch("torch.utils.cpp_extension.CUDA_HOME", None),
        ):
            with pytest.raises(RuntimeError, match="requires `nvcc`"):
                check_toolchain("cuda")

    def test_xpu_ok_when_available(self):
        fake_xpu = SimpleNamespace(is_available=lambda: True)
        with patch("neml2._accelerator.torch.xpu", fake_xpu, create=True):
            check_toolchain("xpu")  # must not raise

    def test_xpu_raises_when_unavailable(self):
        fake_xpu = SimpleNamespace(is_available=lambda: False)
        with patch("neml2._accelerator.torch.xpu", fake_xpu, create=True):
            with pytest.raises(RuntimeError, match="XPU AOTI export requires"):
                check_toolchain("xpu")

    def test_hip_raises_when_unavailable(self):
        # No `torch.hip` attribute → check fails cleanly, no AttributeError.
        with patch("neml2._accelerator.torch", SimpleNamespace()):
            with pytest.raises(RuntimeError, match="HIP AOTI export requires"):
                check_toolchain("hip")

    def test_mps_raises_when_unavailable(self):
        fake_mps = SimpleNamespace(is_available=lambda: False)
        fake_backends = SimpleNamespace(mps=fake_mps)
        with patch("neml2._accelerator.torch", SimpleNamespace(backends=fake_backends)):
            with pytest.raises(RuntimeError, match="MPS AOTI export requires"):
                check_toolchain("mps")
