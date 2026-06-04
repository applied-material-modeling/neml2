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

"""Native ``TransientDriver`` — mirror of C++ ``src/neml2/drivers/TransientDriver.cxx``.

Steps a model forward in time per a prescribed time history + driving-force
arrays. Builds an in-memory ``input.<step>.<var>`` / ``output.<step>.<var>``
result dict (raw ``torch.Tensor`` values) ready for ``TransientRegression`` to
diff against a C++-generated ``gold/result.pt``.

Key differences from C++:
- Step 0 stores only the prescribed forces (time + driving forces); the model
  is NOT called.
- Steps 1+ call ``model(*ordered_inputs)`` once; history (``X~k``) inputs are
  populated from the previous step's outputs (or ICs at step 0).
- Inputs not set by force/IC/history (e.g. ``t~1`` at step 0) default to
  zero — matches the C++ "handled by the forward operator" convention.
- ``save_as`` is parsed but ignored; results stay in memory and are read by
  ``TransientRegression`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nmhit
import torch

from ..driver import Driver
from ..factory import register_native
from ..schema import HitField, HitSchema, dependency, option
from ..types import R2, SR2, SSR4, WR2, MillerIndex, Rot, Scalar, TensorWrapper, Vec

if TYPE_CHECKING:
    from ..factory import _NativeInputFile
    from ..model import Model

#: HIT ``force_<Type>_*`` / ``ic_<Type>_*`` tag -> native wrapper class. Mirrors
#: ``testing._TYPE_MAP`` (and the C++ ``FOR_ALL_TENSORBASE`` macro expansion).
_TYPE_MAP: dict[str, type[TensorWrapper]] = {
    "Scalar": Scalar,
    "Vec": Vec,
    "SR2": SR2,
    "WR2": WR2,
    "R2": R2,
    "Rot": Rot,
    "SSR4": SSR4,
    "MillerIndex": MillerIndex,
}


def _resolve_typed_value(
    factory: _NativeInputFile, type_cls: type[TensorWrapper], token: str
) -> TensorWrapper:
    """Resolve one ``..._values`` token to a typed wrapper.

    Inline ``Scalar`` literals are accepted (the C++ ``TensorName<T>`` convention);
    everything else is a ``[Tensors]`` block name.
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


def _read_typed_pairs(
    driver: nmhit.Node, factory: _NativeInputFile, prefix: str
) -> dict[str, TensorWrapper]:
    """Read ``{prefix}_<Type>_names`` / ``{prefix}_<Type>_values`` into a dict."""
    out: dict[str, TensorWrapper] = {}
    for type_name, type_cls in _TYPE_MAP.items():
        names_key = f"{prefix}_{type_name}_names"
        if driver.find(names_key) is None:
            continue
        names = driver.param_list_str(names_key)
        if not names:
            continue
        values = driver.param_list_str(f"{prefix}_{type_name}_values")
        for var_name, token in zip(names, values, strict=True):
            out[var_name] = _resolve_typed_value(factory, type_cls, token)
    return out


def _typed_io_fields() -> tuple[HitField, ...]:
    """Build documentation fields for the ``force_<Type>_*`` / ``ic_<Type>_*``
    name/value pairs that ``_read_typed_pairs`` ingests at parse time.

    Each typed wrapper (Scalar, Vec, SR2, ...) maps to four optional list
    options: ``force_<T>_names`` paired with ``force_<T>_values`` and the same
    pair under ``ic_<T>_*`` for initial conditions. The schema doesn't drive
    parsing here (the driver overrides ``from_hit``) — it exists so
    ``neml2-syntax`` can render the full HIT surface.
    """
    fields: list[HitField] = []
    for type_name in _TYPE_MAP:
        for prefix, prefix_doc, article in (
            ("force", "driving force", "a"),
            ("ic", "initial condition", "an"),
        ):
            fields.append(
                option(
                    f"{prefix}_{type_name}_names",
                    list,
                    f"{type_name} variable names assigned {article} {prefix_doc} value.",
                    default=[],
                )
            )
            fields.append(
                option(
                    f"{prefix}_{type_name}_values",
                    list,
                    f"{type_name} {prefix_doc} tokens. Each is either an inline Scalar literal "
                    f"(Scalar only) or a [Tensors] block name. Length must match "
                    f"``{prefix}_{type_name}_names``.",
                    default=[],
                )
            )
    return tuple(fields)


@register_native("TransientDriver")
class TransientDriver(Driver):
    """Drive a model over a prescribed time history.

    Mirrors C++ ``TransientDriver``. See module docstring for differences.
    """

    # Documentation-only schema; ``from_hit`` below owns the parsing because
    # ``force_<Type>_*`` / ``ic_<Type>_*`` are a wide product of optional list
    # options that the typed Model-style schema can't express directly.
    hit = HitSchema(
        dependency("model", "get_model", "The Model to drive over the time history."),
        option(
            "prescribed_time",
            str,
            "[Tensors] block name of a Scalar whose leading axis is the per-step time history.",
        ),
        option(
            "time",
            str,
            "Name of the model input that receives the per-step time value.",
            default="t",
        ),
        option(
            "save_as",
            str,
            "Output file path. Accepted for compatibility with the C++ driver but ignored "
            "by the native runtime; results are held in memory and retrieved via "
            ":meth:`result`.",
            default="",
        ),
        *_typed_io_fields(),
    )

    def __init__(
        self,
        model: Model,
        prescribed_time: Scalar,
        time_name: str,
        forces: dict[str, TensorWrapper],
        ics: dict[str, TensorWrapper],
    ) -> None:
        self.model = model
        self.prescribed_time = prescribed_time
        self.time_name = time_name
        self.forces = forces
        self.ics = ics
        self.nsteps = int(prescribed_time.data.shape[0])
        # Per-step input/output dicts, populated by run(). Values are typed
        # wrappers (TensorWrapper) so sub_batch_ndim propagates; the public
        # result() unwraps to raw torch.Tensor at the boundary.
        self.result_in: list[dict[str, TensorWrapper]] = []
        self.result_out: list[dict[str, TensorWrapper]] = []
        self._validate()

    def _validate(self) -> None:
        # Every force should be present in input_spec (or it would never be
        # picked up); same for IC names that match an output_spec key.
        spec = self.model.input_spec
        for name in self.forces:
            if name not in spec:
                raise ValueError(
                    f"TransientDriver force {name!r} is not in model.input_spec {list(spec)}"
                )
        # ICs may reference history vars (input_spec) or step-0 outputs (output_spec).
        for name in self.ics:
            if name not in spec and name not in self.model.output_spec:
                raise ValueError(
                    f"TransientDriver IC {name!r} is in neither input_spec nor "
                    f"output_spec of {type(self.model).__name__}"
                )
        # Driving forces must carry a leading time-step dim of the same size as
        # prescribed_time's leading dim.
        for name, force in self.forces.items():
            if force.data.shape[0] != self.nsteps:
                raise ValueError(
                    f"TransientDriver force {name!r} has leading shape "
                    f"{force.data.shape[0]} but prescribed_time has {self.nsteps}"
                )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> TransientDriver:
        model_name = node.param_str("model")
        model = factory.get_model(model_name)
        time_name = node.param_optional_str("time", "t")
        prescribed_time_name = node.param_str("prescribed_time")
        prescribed_time = factory.get_tensor(prescribed_time_name)
        if not isinstance(prescribed_time, Scalar):
            if isinstance(prescribed_time, torch.Tensor):
                prescribed_time = Scalar(prescribed_time)
            else:
                raise TypeError(
                    f"[Tensors/{prescribed_time_name}] produced "
                    f"{type(prescribed_time).__name__}, expected Scalar"
                )

        forces = _read_typed_pairs(node, factory, "force")
        ics = _read_typed_pairs(node, factory, "ic")

        # save_as is parsed but ignored; native holds results in memory.
        _ = node.param_optional_str("save_as", "")

        return cls(
            model=model,
            prescribed_time=prescribed_time,
            time_name=time_name,
            forces=forces,
            ics=ics,
        )

    def run(self) -> bool:
        spec = self.model.input_spec
        # Per-step input/output dicts hold typed wrappers so sub_batch_ndim
        # propagates through the time loop (raw .data would lose it). The
        # public result() unwraps to raw torch.Tensor at the boundary.
        self.result_in = [{} for _ in range(self.nsteps)]
        self.result_out = [{} for _ in range(self.nsteps)]

        for step in range(self.nsteps):
            cur_in = self.result_in[step]

            # Time + driving forces always come from the prescribed arrays.
            # Slice the typed wrapper so sub_batch_ndim is preserved per step.
            if self.time_name in spec:
                cur_in[self.time_name] = _slice_typed(self.prescribed_time, step)
            for fname, fval in self.forces.items():
                cur_in[fname] = _slice_typed(fval, step)

            # ICs apply only at step 0. history-bearing IC keys (``X~k``) land
            # in input; non-history IC keys (``X`` matching an output_spec key)
            # land in step-0 output.
            if step == 0:
                for ic_name, ic_val in self.ics.items():
                    if ic_name in spec:
                        cur_in[ic_name] = ic_val
                    else:  # in output_spec
                        self.result_out[0][ic_name] = ic_val

            # advance_step: copy previous step's outputs into current step's
            # history inputs (X~k from prev X~(k-1) or prev X for k=1).
            if step > 0:
                prev_in = self.result_in[step - 1]
                prev_out = self.result_out[step - 1]
                for vname in spec:
                    base, order = _parse_history(vname)
                    if order == 0:
                        continue
                    prev_vname = base if order == 1 else f"{base}~{order - 1}"
                    if prev_vname in prev_out:
                        cur_in[vname] = prev_out[prev_vname]
                    elif prev_vname in prev_in:
                        cur_in[vname] = prev_in[prev_vname]

            # Step 0: don't call the model (matches C++ apply_ic-only path).
            if step == 0:
                continue

            # Build the model-call args. Inputs explicitly populated by
            # forces / IC / advance_step come from cur_in (which is also the
            # serialized result_in[step]). Missing inputs default to zeros
            # sized to the per-step batch — needed so the equation system's
            # initial-unknown shapes match the per-step residual shapes that
            # Newton expects — but these defaults stay OUT of cur_in so the
            # serialized output mirrors C++ (which only records
            # explicitly-provided values, leaving zero-defaults to the
            # model's forward operator).
            time_slice = self.prescribed_time.data[step]

            # Missing model inputs default to BASE-SHAPE zeros (no per-step
            # batch). Broadcasting handles per-step batch during forward;
            # this mirrors C++ ``VariableStore::zero_undefined_input``, which
            # uses ``VariableBase::zeros(options)`` (base-shape only). When a
            # Newton solve converges in 0 iterations (initial guess satisfies
            # the residual), the unknown stays at the base-shape initial
            # guess — matching the C++ gold's ``output.<k>.flow_rate`` shape
            # of ``()`` rather than ``(per_step_batch,)``.
            def _default(
                name: str,
                _dtype: torch.dtype = time_slice.dtype,
                _device: torch.device = time_slice.device,
            ) -> TensorWrapper:
                zero_data = _zero_for_step(spec[name], (), _dtype, _device)
                return spec[name](zero_data)

            ordered = tuple(
                _as_typed(cur_in[name], spec[name]) if name in cur_in else _default(name)
                for name in spec
            )
            outs = self.model(*ordered)
            if not isinstance(outs, tuple):
                outs = (outs,)
            if len(outs) != len(self.model.output_spec):
                raise AssertionError(
                    f"{type(self.model).__name__} returned {len(outs)} outputs, "
                    f"expected {len(self.model.output_spec)}"
                )
            for oname, ovalue in zip(self.model.output_spec, outs, strict=True):
                # Keep typed wrappers in result_out so the next step's
                # advance_step preserves sub_batch_ndim.
                self.result_out[step][oname] = ovalue

        return True

    def result(self) -> dict[str, torch.Tensor]:
        """Flatten ``result_in`` / ``result_out`` into a ``input.<i>.<name>``
        / ``output.<i>.<name>`` dict, matching the C++ ``ModuleDict`` layout
        that ``torch.jit.load(gold).named_buffers()`` produces. Typed wrappers
        are unwrapped to raw ``torch.Tensor`` at this boundary.
        """
        out: dict[str, torch.Tensor] = {}
        for step, step_in in enumerate(self.result_in):
            for name, val in step_in.items():
                out[f"input.{step}.{name}"] = val.data if isinstance(val, TensorWrapper) else val
        for step, step_out in enumerate(self.result_out):
            for name, val in step_out.items():
                out[f"output.{step}.{name}"] = val.data if isinstance(val, TensorWrapper) else val
        return out

    def save_gold(self, path: str | Path) -> None:
        """Serialize ``result()`` to a ``.pt`` file as a flat ``torch.save`` dict.

        The dict carries the same keys as :meth:`result`:
        ``input.<step>.<var>`` / ``output.<step>.<var>``, each mapping to a
        detached, cloned ``torch.Tensor``. Read it back with
        ``torch.load(path, weights_only=True)``.

        Replaces the old TorchScript-Module format (a Module-of-Modules with
        ``register_buffer`` per entry, dumped via
        ``torch.jit.script(...).save(...)``). The flat-dict format avoids the
        ``torch.jit.{script,load}`` deprecation warnings and survives
        round-trip with ``weights_only=True`` (which is safe for plain tensor
        dicts and prevents arbitrary code execution on load).
        ``TransientRegression`` reads both formats so existing on-disk goldens
        from the v2 C++ pipeline keep working — see
        :class:`~neml2.drivers.TransientRegression.TransientRegression`.
        """
        torch.save(self.result(), str(Path(path)))


def _slice_typed(wrapped: TensorWrapper, step: int) -> TensorWrapper:
    """Slice the leading time-step dim and re-wrap, preserving sub_batch_ndim."""
    return type(wrapped)(wrapped.data[step], sub_batch_ndim=wrapped.sub_batch_ndim)


def _as_typed(value, type_cls: type[TensorWrapper]) -> TensorWrapper:
    """Coerce ``value`` to ``type_cls``, preserving sub_batch_ndim when possible."""
    if isinstance(value, type_cls):
        return value
    if isinstance(value, TensorWrapper):
        return type_cls(value.data, sub_batch_ndim=value.sub_batch_ndim)
    return type_cls(value)


def _parse_history(vname: str) -> tuple[str, int]:
    """Split ``base~k`` -> ``(base, k)``; bare ``base`` -> ``(base, 0)``."""
    if "~" not in vname:
        return vname, 0
    base, suffix = vname.rsplit("~", 1)
    try:
        return base, int(suffix)
    except ValueError:
        return vname, 0


def _zero_for_step(
    type_cls: type[TensorWrapper],
    per_step_batch_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Make a zero ``torch.Tensor`` shaped ``(*per_step_batch_shape, *base_shape)``.

    ``per_step_batch_shape`` is read from the prescribed_time slice at the
    current step — a Scalar with ``base_shape=()``, so its data shape IS the
    per-step batch shape. Each typed wrapper adds its own base axes.
    """
    base_shape = tuple(getattr(type_cls, "BASE_SHAPE", ()))
    return torch.zeros(per_step_batch_shape + base_shape, dtype=dtype, device=device)


__all__ = ["TransientDriver"]
