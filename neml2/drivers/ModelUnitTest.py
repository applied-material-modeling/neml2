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

"""Testing helpers for Python-native models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nmhit
import torch

from ..factory import load_input, load_string
from ..models._guard import allow_autograd
from ..models.chain_rule import ChainRuleDict
from ..models.model import Model
from ..types import MRP, R2, SR2, SSR4, WR2, MillerIndex, Scalar, TensorWrapper, Vec

if TYPE_CHECKING:
    from ..factory import _NativeInputFile

__all__ = ["ModelUnitTest", "ModelUnitTestReport"]

#: HIT ``input_<Type>_*`` / ``output_<Type>_*`` tag -> native wrapper class.
_TYPE_MAP: dict[str, type[TensorWrapper]] = {
    "Scalar": Scalar,
    "Vec": Vec,
    "SR2": SR2,
    "WR2": WR2,
    "R2": R2,
    "MRP": MRP,
    "SSR4": SSR4,
    "MillerIndex": MillerIndex,
}


@dataclass(frozen=True)
class ModelUnitTestReport:
    """Summary returned by :meth:`ModelUnitTest.run`."""

    value_checks: int
    jvp_checks: int


def _as_tuple(value) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def _rewrap(template: TensorWrapper, data: torch.Tensor) -> TensorWrapper:
    return type(template)(data, sub_batch_ndim=template.sub_batch_ndim)


def _default_tangent(x: TensorWrapper) -> TensorWrapper:
    data = x.data
    if not data.is_floating_point():
        raise TypeError(f"ModelUnitTest inputs must be floating-point tensors, got {data.dtype}")
    n = data.numel()
    tangent = torch.arange(1, n + 1, dtype=data.dtype, device=data.device).reshape(data.shape)
    tangent = tangent / (n + 1)
    return _rewrap(x, tangent)


class ModelUnitTest:
    """Harness for native model value and first-derivative checks.

    The preferred entry points are the input-file constructors
    :meth:`from_string` (inline HIT) and :meth:`from_file` (a ``.i`` file): they
    build the model and its typed inputs/expected outputs from a ``[Drivers]``
    ``type = ModelUnitTest`` block, mirroring the C++ ``ModelUnitTest`` so the
    same scenarios drive both backends. The lower-level constructor that takes a
    pre-built ``model`` plus Python ``inputs`` is still available. First
    derivatives are checked as directional JVPs by comparing the model's
    analytical ``forward(v=...)`` pushforward against
    ``torch.autograd.functional.jvp`` of the same eager forward.
    """

    def __init__(
        self,
        model: Model,
        inputs: Mapping[str, TensorWrapper],
        *,
        expected_outputs: Mapping[str, TensorWrapper | torch.Tensor] | None = None,
        tangents: Mapping[str, TensorWrapper | Sequence[TensorWrapper]] | None = None,
        rtol: float = 1.0e-10,
        atol: float = 1.0e-12,
        derivative_rtol: float | None = None,
        derivative_atol: float | None = None,
    ) -> None:
        self.model = model
        self.inputs = dict(inputs)
        self.expected_outputs = dict(expected_outputs or {})
        self.tangents = dict(tangents or {})
        self.rtol = rtol
        self.atol = atol
        # Derivative checks can carry looser tolerances (the C++ ModelUnitTest
        # exposes ``derivative_rel_tol`` / ``derivative_abs_tol`` separately);
        # default to the value tolerances when not specified.
        self.drtol = rtol if derivative_rtol is None else derivative_rtol
        self.datol = atol if derivative_atol is None else derivative_atol
        self._validate_inputs()

    @classmethod
    def from_string(cls, text: str, *, name: str | None = None) -> ModelUnitTest:
        """Build a unit test from an in-memory HIT snippet (input-file-only path).

        The snippet must contain a ``[Drivers]`` block of ``type =
        ModelUnitTest`` (plus the ``[Models]`` it names and any ``[Tensors]`` it
        references). The model is built from the ``[Models]`` block via the
        native factory; inputs and expected outputs are read from the driver's
        ``input_<Type>_names`` / ``input_<Type>_values`` (and ``output_<Type>_*``)
        options. ``value_rel_tol`` / ``value_abs_tol`` override the tolerances
        (defaults mirror the C++ ``ModelUnitTest``: ``1e-5`` / ``1e-8``).
        """
        return cls._from_factory(load_string(text), name=name)

    @classmethod
    def from_file(cls, path: str | Path, *, name: str | None = None) -> ModelUnitTest:
        """Build a unit test from a HIT ``.i`` file. See :meth:`from_string`."""
        return cls._from_factory(load_input(Path(path)), name=name)

    @classmethod
    def _from_factory(cls, factory: _NativeInputFile, *, name: str | None) -> ModelUnitTest:
        driver = _find_unit_test_driver(factory, name)
        model = factory.get_model(driver.param_str("model"))
        inputs = _read_typed_io(driver, factory, "input")
        expected = _read_typed_io(driver, factory, "output")
        rtol = driver.param_optional_float("value_rel_tol", 1.0e-5)
        atol = driver.param_optional_float("value_abs_tol", 1.0e-8)
        drtol = driver.param_optional_float("derivative_rel_tol", 1.0e-5)
        datol = driver.param_optional_float("derivative_abs_tol", 1.0e-8)
        return cls(
            model,
            inputs,
            expected_outputs=expected,
            rtol=rtol,
            atol=atol,
            derivative_rtol=drtol,
            derivative_atol=datol,
        )

    def _validate_inputs(self) -> None:
        missing = [name for name in self.model.input_spec if name not in self.inputs]
        extra = [name for name in self.inputs if name not in self.model.input_spec]
        if missing or extra:
            raise ValueError(
                f"ModelUnitTest input mismatch for {type(self.model).__name__}: "
                f"missing={missing}, extra={extra}"
            )
        for name, type_cls in self.model.input_spec.items():
            if not isinstance(self.inputs[name], type_cls):
                raise TypeError(
                    f"input {name!r} must be {type_cls.__name__}, "
                    f"got {type(self.inputs[name]).__name__}"
                )
        for name in self.expected_outputs:
            if name not in self.model.output_spec:
                raise ValueError(f"expected output {name!r} is not in model.output_spec")

    def _ordered_inputs(self) -> tuple[TensorWrapper, ...]:
        return tuple(self.inputs[name] for name in self.model.input_spec)

    def _forward_outputs(self) -> tuple[TensorWrapper, ...]:
        outputs = _as_tuple(self.model(*self._ordered_inputs()))
        if len(outputs) != len(self.model.output_spec):
            raise AssertionError(
                f"{type(self.model).__name__} returned {len(outputs)} outputs, "
                f"expected {len(self.model.output_spec)}"
            )
        return outputs

    def check_values(self) -> int:
        """Check forward values against ``expected_outputs``.

        Returns the number of output tensors checked. If no expected outputs
        were supplied, this is a no-op and returns ``0``.
        """

        if not self.expected_outputs:
            return 0
        outputs = dict(zip(self.model.output_spec, self._forward_outputs(), strict=True))
        for name, expected in self.expected_outputs.items():
            actual = outputs[name].data
            expected_data = expected.data if isinstance(expected, TensorWrapper) else expected
            torch.testing.assert_close(
                actual,
                expected_data,
                rtol=self.rtol,
                atol=self.atol,
                msg=f"value mismatch for output {name!r}",
            )
        return len(self.expected_outputs)

    def _input_tangents(self, name: str) -> tuple[TensorWrapper, ...]:
        supplied = self.tangents.get(name)
        if supplied is None:
            return (_default_tangent(self.inputs[name]),)
        if isinstance(supplied, TensorWrapper):
            return (supplied,)
        return tuple(supplied)

    def _analytical_jvp(self, input_name: str, tangent: TensorWrapper) -> dict[str, TensorWrapper]:
        v: ChainRuleDict = {input_name: {input_name: tangent}}
        result = _as_tuple(self.model(*self._ordered_inputs(), v=v))
        if len(result) != len(self.model.output_spec) + 1:
            raise AssertionError(
                f"{type(self.model).__name__}.forward(v=...) returned {len(result)} items, "
                f"expected {len(self.model.output_spec) + 1}"
            )
        v_out = result[-1]
        return {
            out_name: v_out.get(out_name, {}).get(
                input_name,
                type_cls(torch.zeros_like(value.data), sub_batch_ndim=value.sub_batch_ndim),
            )
            for (out_name, type_cls), value in zip(
                self.model.output_spec.items(), result[:-1], strict=True
            )
        }

    def _autograd_jvp(self, input_name: str, tangent: TensorWrapper) -> dict[str, torch.Tensor]:
        input_names = list(self.model.input_spec)
        idx = input_names.index(input_name)
        base_inputs = list(self._ordered_inputs())
        primal = base_inputs[idx].data.detach().clone().requires_grad_(True)
        tangent_data = tangent.data

        def fn(x_data: torch.Tensor) -> tuple[torch.Tensor, ...]:
            args = list(base_inputs)
            args[idx] = _rewrap(args[idx], x_data)
            outputs = _as_tuple(self.model(*args))
            return tuple(out.data for out in outputs)

        with allow_autograd("native ModelUnitTest derivative oracle"):
            _, ad_tangent = torch.autograd.functional.jvp(
                fn,
                (primal,),
                (tangent_data,),
                create_graph=False,
                strict=False,
            )
        ad_tangent = _as_tuple(ad_tangent)
        return dict(zip(self.model.output_spec, ad_tangent, strict=True))

    def check_dvalue(self) -> int:
        """Check first derivatives using directional JVPs.

        Returns the number of ``(input tangent, output)`` comparisons made.
        """

        checks = 0
        for input_name, type_cls in self.model.input_spec.items():
            for tangent in self._input_tangents(input_name):
                if not isinstance(tangent, type_cls):
                    raise TypeError(
                        f"tangent for input {input_name!r} must be {type_cls.__name__}, "
                        f"got {type(tangent).__name__}"
                    )
                analytical = self._analytical_jvp(input_name, tangent)
                oracle = self._autograd_jvp(input_name, tangent)
                for output_name in self.model.output_spec:
                    torch.testing.assert_close(
                        analytical[output_name].data,
                        oracle[output_name],
                        rtol=self.drtol,
                        atol=self.datol,
                        msg=f"JVP mismatch for d({output_name})/d({input_name})",
                    )
                    checks += 1
        return checks

    def run(self, *, check_values: bool = True, check_dvalue: bool = True) -> ModelUnitTestReport:
        """Run enabled checks and return a count summary."""

        value_checks = self.check_values() if check_values else 0
        jvp_checks = self.check_dvalue() if check_dvalue else 0
        return ModelUnitTestReport(value_checks=value_checks, jvp_checks=jvp_checks)


def _find_unit_test_driver(factory: _NativeInputFile, name: str | None) -> nmhit.Node:
    """Locate the ``[Drivers]`` sub-block of ``type = ModelUnitTest``."""
    for top in factory._root.children(nmhit.NodeType.Section):
        if top.path() != "Drivers":
            continue
        for child in top.children(nmhit.NodeType.Section):
            if name is not None and child.path().rsplit("/", 1)[-1] != name:
                continue
            if child.param_optional_str("type", "") == "ModelUnitTest":
                return child
    raise ValueError(
        "ModelUnitTest input must contain a [Drivers] block of type 'ModelUnitTest'"
        + (f" named {name!r}" if name is not None else "")
    )


def _resolve_typed_value(
    factory: _NativeInputFile, type_cls: type[TensorWrapper], token: str
) -> TensorWrapper:
    """Resolve one ``..._values`` token to a typed wrapper.

    A token is either an inline ``Scalar`` literal (``'0.1'``) or the name of a
    ``[Tensors]`` block (``'NX'``) resolved through the factory — mirroring the
    C++ ``TensorName<T>`` semantics used by the ``ModelUnitTest`` driver.
    """
    try:
        literal = float(token)
    except ValueError:
        literal = None
    if literal is not None:
        if type_cls is not Scalar:
            raise ValueError(
                f"inline literal {token!r} is only supported for Scalar; "
                f"reference a [Tensors] block for {type_cls.__name__} values"
            )
        return Scalar(torch.tensor(literal, dtype=torch.float64))

    value = factory.get_tensor(token)
    if isinstance(value, type_cls):
        return value
    if isinstance(value, torch.Tensor):
        return type_cls(value)
    raise TypeError(
        f"[Tensors/{token}] produced {type(value).__name__}, expected {type_cls.__name__}"
    )


def _read_typed_io(
    driver: nmhit.Node, factory: _NativeInputFile, prefix: str
) -> dict[str, TensorWrapper]:
    """Read ``{prefix}_<Type>_names`` / ``{prefix}_<Type>_values`` into a dict."""
    result: dict[str, TensorWrapper] = {}
    for type_name, type_cls in _TYPE_MAP.items():
        names_key = f"{prefix}_{type_name}_names"
        if driver.find(names_key) is None:
            continue
        names = driver.param_list_str(names_key)
        values = driver.param_list_str(f"{prefix}_{type_name}_values")
        for var_name, token in zip(names, values, strict=True):
            result[var_name] = _resolve_typed_value(factory, type_cls, token)
    return result
