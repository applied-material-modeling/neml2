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

"""Argparse-level device parsing for neml2-compile and neml2-run.

Verifies that the accelerator-family generalization (XPU/HIP/MPS in addition
to CPU/CUDA) is wired at the CLI without breaking the existing CPU/CUDA
behavior. These tests never invoke the compile step; they only exercise the
argument parser and the shared ``_accelerator`` validator, so they run on any
host regardless of installed torch backends.
"""

from __future__ import annotations

import pytest

from neml2._accelerator import KNOWN_FAMILIES


class TestNeml2CompileDeviceChoices:
    def test_default_is_cpu(self):
        from neml2.cli.aoti_compile import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(["input.i", "--model", "m"])
        assert args.device == ["cpu"]

    @pytest.mark.parametrize("fam", KNOWN_FAMILIES)
    def test_each_family_is_accepted(self, fam):
        from neml2.cli.aoti_compile import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(["input.i", "--model", "m", "--device", fam])
        assert args.device == [fam]

    def test_multiple_devices_accepted(self):
        from neml2.cli.aoti_compile import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(["input.i", "--model", "m", "--device", "cpu", "cuda", "xpu"])
        assert args.device == ["cpu", "cuda", "xpu"]

    def test_rejects_unknown_family(self):
        from neml2.cli.aoti_compile import _build_arg_parser

        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["input.i", "--model", "m", "--device", "foo"])

    def test_rejects_indexed_form_on_compile(self):
        # `neml2-compile` targets a device *family* (one binary per family), so
        # indexed forms are not accepted here -- argparse's `choices` enforces
        # bare family names. Indexed forms belong to `neml2-run` (runtime pin).
        from neml2.cli.aoti_compile import _build_arg_parser

        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["input.i", "--model", "m", "--device", "cuda:0"])


class TestNeml2RunDeviceParsing:
    def test_default_is_cpu(self):
        from neml2.cli.run import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["input.i", "driver"])
        assert args.device == "cpu"

    @pytest.mark.parametrize("fam", KNOWN_FAMILIES)
    def test_each_family_is_accepted(self, fam):
        from neml2.cli.run import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["input.i", "driver", "--device", fam])
        assert args.device == fam

    @pytest.mark.parametrize("spec", ["cuda:0", "cuda:1", "xpu:0"])
    def test_indexed_forms_accepted_on_run(self, spec):
        # `neml2-run` sets `torch.set_default_device`, so a specific device
        # index is meaningful and accepted.
        from neml2.cli.run import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["input.i", "driver", "--device", spec])
        assert args.device == spec

    def test_rejects_unknown_family(self, capsys):
        from neml2.cli.run import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["input.i", "driver", "--device", "foo"])
        # argparse catches the ValueError from `_device_arg` and formats its
        # own "invalid <type> value" line; the underlying validator's message
        # is swallowed. We just verify the invalid-value message reaches the
        # user (either stream is fine).
        captured = capsys.readouterr()
        assert "invalid _device_arg value: 'foo'" in (captured.out + captured.err)


class TestAotiShimDeviceValidation:
    def test_rejects_unknown_family(self, tmp_path):
        # A minimum valid artifact_root: contains a metadata.json so the shim
        # gets past the FileNotFoundError guard and reaches the device check.
        (tmp_path / "metadata.json").write_text('{"schema_version": 14}')
        from neml2.aoti._shim import AOTIModel

        with pytest.raises(ValueError, match="unknown device family"):
            AOTIModel(tmp_path, device="foo")


class TestAcceleratorJobsWarning:
    """The `_j > 1 with an accelerator target` warning printed by neml2-compile.

    Extracted from `main()` as `_accelerator_jobs_warning` so the branch is
    unit-testable without spinning up a real compile pool.
    """

    def test_returns_none_for_serial(self):
        from neml2.cli.aoti_compile import _accelerator_jobs_warning

        assert _accelerator_jobs_warning(1, ["cpu", "cuda"]) is None

    def test_returns_none_for_cpu_only_parallel(self):
        from neml2.cli.aoti_compile import _accelerator_jobs_warning

        # Multiple workers on CPU alone is fine -- the warning is only about
        # accelerator contexts + codegen compilers.
        assert _accelerator_jobs_warning(4, ["cpu"]) is None

    def test_warns_for_parallel_cuda(self):
        from neml2.cli.aoti_compile import _accelerator_jobs_warning

        msg = _accelerator_jobs_warning(4, ["cpu", "cuda"])
        assert msg is not None
        assert "-j4" in msg
        assert "cuda" in msg

    def test_warns_for_parallel_multi_accelerator(self):
        from neml2.cli.aoti_compile import _accelerator_jobs_warning

        msg = _accelerator_jobs_warning(2, ["cpu", "cuda", "xpu"])
        assert msg is not None
        # Families listed sorted, cpu omitted.
        assert "cuda, xpu" in msg
