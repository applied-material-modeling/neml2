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
    is_compatible,
    parse_device_spec,
    partition_compatible_devices,
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

    def test_rejects_wrong_type_input(self):
        # ``torch.device(None)`` raises TypeError (not RuntimeError). Cover the
        # second branch of the ``except (RuntimeError, TypeError)`` clause so
        # a non-string spec still surfaces the neml2 known-families message.
        with pytest.raises(ValueError, match="unknown device family"):
            parse_device_spec(None)  # type: ignore[arg-type]


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

    def test_meta_is_non_accelerator(self):
        # ``meta`` is torch's shape-only device; it needs no toolchain preflight
        # and the C++ runtime doesn't route it as an accelerator. Treat it like
        # cpu (skip; return None if no real accelerator follows).
        t = torch.zeros(2, device="meta")
        assert target_accelerator_family(iter([t])) is None

    def test_meta_then_accelerator_still_returns_accelerator(self):
        # A meta tensor doesn't hide a real accelerator that follows.
        from typing import cast

        meta = cast(torch.Tensor, SimpleNamespace(device=torch.device("meta")))
        xpu = cast(torch.Tensor, SimpleNamespace(device=torch.device("xpu")))
        assert target_accelerator_family(iter([meta, xpu])) == "xpu"

    def test_first_accelerator_wins(self):
        # Fake tensor-like with .device.type via SimpleNamespace, so the test
        # runs without any real accelerator. The function's ordering contract
        # is what we're testing; it only reads .device.type.
        from typing import cast

        cpu = cast(torch.Tensor, SimpleNamespace(device=torch.device("cpu")))
        xpu = cast(torch.Tensor, SimpleNamespace(device=torch.device("xpu")))
        cuda = cast(torch.Tensor, SimpleNamespace(device=torch.device("cuda")))
        assert target_accelerator_family(iter([cpu, xpu, cuda])) == "xpu"

    def test_unknown_family_raises(self):
        # A tensor on a family NEML2 doesn't know how to route (not cpu/meta
        # and not in KNOWN_FAMILIES) is a bug -- the C++ runtime cannot
        # dispatch to it. Fail loudly at the front instead of letting the
        # compile continue to a cryptic ValueError inside check_toolchain.
        from typing import cast

        # Torch doesn't let us construct a device object for an entirely
        # unknown family (`torch.device("nvptx")` raises), so stub the tensor
        # with a device-like object exposing `.type` alone.
        weird = cast(torch.Tensor, SimpleNamespace(device=SimpleNamespace(type="nvptx")))
        with pytest.raises(ValueError, match="not one NEML2 knows how to route"):
            target_accelerator_family(iter([weird]))


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


class TestExportPreflightBranch:
    """Cover the `if _target_family is not None:` accelerator-preflight branch
    in `neml2.models.export.compile_model` -- CPU-only test paths never
    exercise it because every example input lives on CPU, and meta is a
    non-accelerator so it doesn't trigger the branch either.

    Monkey-patch ``target_accelerator_family`` to claim any input is on an
    accelerator, and ``check_toolchain`` to raise a sentinel that short-
    circuits before the real Inductor compile. That proves the branch fired
    without needing accelerator hardware.
    """

    def test_accelerator_example_input_triggers_toolchain_check(self, tmp_path, monkeypatch):
        # ``compile_model`` imports ``target_accelerator_family`` and
        # ``check_toolchain`` deferred, so patch the source module -- the
        # deferred ``from`` import will pick up the patched attribute.
        import torch.nn as nn

        from neml2.models.export import compile_model

        class Identity(nn.Module):
            def forward(self, x):
                return x

        called: list[str] = []

        def _fake_check(fam: str) -> None:
            called.append(fam)
            raise RuntimeError(f"stop-here: {fam}")

        monkeypatch.setattr("neml2._accelerator.target_accelerator_family", lambda _: "xpu")
        monkeypatch.setattr("neml2._accelerator.check_toolchain", _fake_check)

        with pytest.raises(RuntimeError, match="stop-here: xpu"):
            compile_model(Identity(), (torch.zeros(2),), tmp_path / "unused.pt2")
        assert called == ["xpu"]

    def test_meta_example_input_does_not_trigger_toolchain_check(self, tmp_path, monkeypatch):
        # The dual: a meta-device tensor is non-accelerator, so the preflight
        # branch must be skipped entirely and ``check_toolchain`` must never run.
        # We short-circuit inside the actual walker (no need to reach Inductor)
        # by having the patched ``check_toolchain`` record any call and then
        # letting the real ``target_accelerator_family`` classify meta as
        # non-accelerator -> returns None -> ``check_toolchain`` is never called.
        import torch.nn as nn

        from neml2.models.export import compile_model

        class Identity(nn.Module):
            def forward(self, x):
                return x

        called: list[str] = []
        monkeypatch.setattr("neml2._accelerator.check_toolchain", lambda fam: called.append(fam))
        # Stub Inductor so we don't actually try to lower a meta tensor; the
        # point is only that the preflight was skipped and control flowed on.
        monkeypatch.setattr(
            "torch._inductor.aoti_compile_and_package",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("compile-reached")),
        )

        with pytest.raises(RuntimeError, match="compile-reached"):
            compile_model(
                Identity(),
                (torch.zeros(2, device="meta"),),
                tmp_path / "unused.pt2",
            )
        assert called == []


class TestIsCompatible:
    @pytest.mark.parametrize(
        "family, dtype, expected",
        [
            ("cpu", "float64", True),
            ("cpu", "float32", True),
            ("cuda", "float64", True),
            ("cuda", "float32", True),
            ("xpu", "float64", True),
            ("xpu", "float32", True),
            ("hip", "float64", True),
            ("hip", "float32", True),
            ("mps", "float32", True),
            ("mps", "float64", False),  # Apple MPS has no fp64 kernel path.
        ],
    )
    def test_known_combinations(self, family, dtype, expected):
        assert is_compatible(family, dtype) is expected

    def test_unknown_family_is_permissive(self):
        # A family not in the restriction table is treated as accepting every
        # dtype. This keeps ``is_compatible`` from fabricating restrictions
        # for accelerators that get added later.
        assert is_compatible("mtia", "float64") is True


class TestPartitionCompatibleDevices:
    def test_all_compatible(self):
        ok, bad = partition_compatible_devices(["cpu", "cuda"], "float64")
        assert ok == ["cpu", "cuda"]
        assert bad == []

    def test_all_incompatible(self):
        ok, bad = partition_compatible_devices(["mps"], "float64")
        assert ok == []
        assert bad == ["mps"]

    def test_mixed(self):
        ok, bad = partition_compatible_devices(["cpu", "mps", "cuda"], "float64")
        assert ok == ["cpu", "cuda"]
        assert bad == ["mps"]

    def test_preserves_order(self):
        ok, bad = partition_compatible_devices(["mps", "cpu", "mps", "cuda"], "float64")
        assert ok == ["cpu", "cuda"]
        assert bad == ["mps", "mps"]

    def test_float32_leaves_mps_valid(self):
        ok, bad = partition_compatible_devices(["cpu", "mps"], "float32")
        assert ok == ["cpu", "mps"]
        assert bad == []
