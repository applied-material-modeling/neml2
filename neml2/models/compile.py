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

"""In-process ``torch.compile`` acceleration for NEML2 feed-forward models.

This is the native-Python counterpart to the AOTI export pipeline in
``neml2.models.export``: instead of lowering a model to a frozen ``.pt2`` package
loaded from a separate (C++) runtime, :func:`compile` JIT-compiles a model's
``forward`` *in the running interpreter* via ``torch.compile``. There is no
artifact, no round-trip, and the compiled graph is reused across subsequent calls
(recompiling lazily only when an input shape signature changes).

The intended consumer is the pyzag training interface
(:class:`neml2.pyzag.NEML2PyzagModel`): pyzag owns the Newton solve and the
adjoint, and only asks NEML2 to evaluate the residual and Jacobian of an explicit
*feed-forward* model. That evaluation -- ``ModelNonlinearSystem.assemble`` calling
the residual ``Model`` with chain-rule JVP seeds -- carries essentially all the
FLOPs and has no data-dependent control flow, so it is an ideal ``torch.compile``
target.

This module is the single containment point for ``torch.compile`` /
``torch._dynamo`` access (mirroring how ``export.py`` contains ``torch._inductor``
for AOTI), so a torch version bump touches one file.

Mechanism: we replace the instance ``forward`` with a ``torch.compile``'d copy of
the original, wrapped in an :func:`~neml2.allow_autograd` window. :meth:`Model.__call__`
still opens the forward guard window and delegates to ``nn.Module.__call__`` ->
``self.forward``, so the guard's enter/exit and the ``allow_autograd`` context stay
in eager code *outside* the compiled graph, and the model object's identity,
``input_spec`` / ``output_spec``, and ``__getattr__`` are untouched. The equation
system reads ``self.model`` live on every call
(:meth:`neml2.es.ModelNonlinearSystem._call_model`), so compiling the residual
model in place needs zero changes to the assembly / solver path, and autograd flows
through the compiled graph transparently (so pyzag's adjoint backward keeps
working).

We compile with ``dynamic=False`` (hard-coded). This gives a single break-free
graph (~3x on the pyzag viscoplastic solve), at the cost of one recompile per
distinct input shape -- acceptable because a fixed training config sees only a
small, stable set of shapes.

TODO (K-state / ``dynamic=True``): NEML2's typed-tensor chain rule threads
*string-valued* K-state metadata (``"full"`` / ``"broadcast"``, plus ``k_ndim`` /
``k_pairing``) through every op (``types/_base.py``: ``align_k``,
``combine_k_state``, ``__post_init__``). That metadata is *structural* -- a function
of the model graph and the chain-rule seeding, not of the dynamic batch dim -- so in
principle it should be a compile-time constant identical across re-traces. But under
``dynamic=True`` Dynamo couples it to the symbolic input shapes (the first graph
break is ``align_k`` reading ``w.k_state`` off a wrapper whose ``.data`` is symbolic;
``__post_init__`` branches on ``self.data.shape[i]``), fragmenting the forward into
~80 churning frames at ~1x. If the metadata is made genuinely Dynamo-constant under
tracing, ``dynamic=True`` should collapse to one clean graph reused across *all*
shapes -- fast *and* shape-general, removing the per-shape recompile. Until then,
``dynamic=False`` is the only setting that actually accelerates, so the knob is not
exposed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import torch

from .model import Model

if TYPE_CHECKING:
    from ..es import ModelNonlinearSystem

__all__ = ["compile"]

_T = TypeVar("_T")


def compile(  # noqa: A001 -- intentionally the public ``neml2.compile`` verb
    target: _T,
    *,
    fullgraph: bool = False,
    mode: str | None = None,
    **compile_kwargs: Any,
) -> _T:
    """Accelerate a NEML2 feed-forward model with ``torch.compile``, in process.

    The ``target`` may be:

    * a :class:`~neml2.models.model.Model` -- its ``forward`` is compiled;
    * a :class:`~neml2.es.ModelNonlinearSystem` -- its residual ``model`` is
      compiled (so ``assemble`` / ``A_and_B_and_b`` run compiled);
    * a pyzag ``NEML2PyzagModel`` (anything exposing a ``.sys`` that is a
      ``ModelNonlinearSystem``) -- the wrapped residual model is compiled.

    Compilation is applied **in place** -- the same object is returned, now
    evaluating through a ``torch.compile``'d graph. Unlike AOTI export, no package
    is produced; the graph lives in the interpreter and recompiles lazily on a new
    input shape signature. Compilation uses ``dynamic=False`` (not exposed; see the
    module-level TODO): it is the only setting that actually accelerates today, and
    it recompiles once per distinct input shape -- fine for a fixed training config,
    whose shape set is small and stable.

    Args:
        target: the model / system / pyzag wrapper to accelerate.

    Keyword Args:
        fullgraph: forwarded to ``torch.compile``. Default ``False`` tolerates
            graph breaks; under ``dynamic=False`` the residual forward already
            compiles into a single break-free graph, so ``True`` also works.
        mode: forwarded to ``torch.compile`` (e.g. ``"reduce-overhead"`` for
            CUDA-graphs, ``"max-autotune"``).
        **compile_kwargs: any other ``torch.compile`` keyword (``backend``,
            ``options``, ...). ``mode`` and ``options`` are mutually exclusive
            in ``torch.compile``; ``dynamic`` is reserved (always ``False``).

    Returns:
        ``target``, compiled in place.

    Raises:
        TypeError: if ``target`` is not a ``Model``, ``ModelNonlinearSystem``, or
            an object wrapping one via ``.sys``.
    """
    # Lazy import: ``neml2.es`` imports ``neml2.models.model`` at module load, so
    # importing it at the top of this file would create a models<->es import
    # cycle. Resolving it inside the function (call time) sidesteps that.
    from ..es import ModelNonlinearSystem  # noqa: PLC0415

    # dynamic=False is hard-coded -- see the module TODO. It is the only setting
    # that produces a clean, accelerated graph today (dynamic=True fragments on the
    # string K-state metadata). Recompiles once per distinct input shape.
    opts: dict[str, Any] = {
        "dynamic": False,
        "fullgraph": fullgraph,
        "mode": mode,
        **compile_kwargs,
    }

    # pyzag NEML2PyzagModel wraps a ModelNonlinearSystem in ``.sys`` and aliases
    # ``.sys.model`` as ``.model`` (same object). Duck-typed on ``.sys`` so we do
    # not import ``neml2.pyzag`` here (it would invert the dependency direction).
    wrapped_system = getattr(target, "sys", None)
    if isinstance(wrapped_system, ModelNonlinearSystem):
        _compile_residual_in_place(wrapped_system, opts)
        return target

    if isinstance(target, ModelNonlinearSystem):
        _compile_residual_in_place(target, opts)
        return target

    if isinstance(target, Model):
        _compile_forward_in_place(target, opts)
        return target

    raise TypeError(
        "neml2.compile expects a neml2 Model, a ModelNonlinearSystem, or an object "
        "wrapping one via `.sys` (e.g. a pyzag NEML2PyzagModel); got "
        f"{type(target).__name__}."
    )


def _compile_residual_in_place(system: ModelNonlinearSystem, opts: dict[str, Any]) -> None:
    """Compile the residual ``Model`` evaluated by ``system`` in place."""
    model = system.model
    if not isinstance(model, Model):
        raise TypeError(
            "ModelNonlinearSystem.model is not a neml2 Model "
            f"(got {type(model).__name__}); cannot compile."
        )
    _compile_forward_in_place(model, opts)


_COMPILED_MARK = "_neml2_compiled_forward"


def _compile_forward_in_place(model: Model, opts: dict[str, Any]) -> None:
    """Compile a ``Model``'s forward in place.

    We replace the instance ``forward`` with ``torch.compile``'d original wrapped in
    an :func:`~neml2.allow_autograd` window. :meth:`Model.__call__` still opens the
    forward guard window and delegates to ``nn.Module.__call__`` -> ``self.forward``,
    so the guard's ``threading.local`` enter/exit and the ``allow_autograd`` context
    both stay in eager code *outside* the compiled graph.

    Why ``allow_autograd``: ``torch.compile``'s backend (AOTAutograd) traces the
    backward of the compiled forward when grad is enabled (e.g. pyzag's
    ``solve_adjoint``), and that tracing calls ``torch.autograd.grad`` internally.
    The forward guard (``_guard.py``) forbids ``torch.autograd.*`` inside a forward
    window because it does not survive *AOTI export* -- but this is the in-process
    ``torch.compile`` path, not the export path, and AOTAutograd handles autograd
    correctly. So we explicitly permit it here (the AOTI export path in
    ``export.py`` is separate and keeps the guard armed). The model body itself is
    still proven banned-op-free by the AOTI export tests.
    """
    from .._warnings import TORCH_JIT_PY314, ignore_warnings  # noqa: PLC0415
    from ._guard import allow_autograd  # noqa: PLC0415

    if getattr(model, _COMPILED_MARK, False):
        return  # already compiled; avoid double-wrapping

    original_forward = model.forward  # bound method, captured before replacement
    # torch.compile eagerly imports torch._inductor.compile_fx (to read the compiler
    # config), which pulls in torch.utils.mkldnn -- and on Python 3.14 that warns
    # about torch's own deprecated torch.jit.script_method (neml2 never calls it).
    # Suppress that one torch-internal line; the spec is shared with the AOTI path
    # via neml2/_warnings.py.
    with ignore_warnings(TORCH_JIT_PY314):
        compiled_forward = torch.compile(original_forward, **opts)

    def _forward(*args: Any, **kwargs: Any) -> Any:
        with allow_autograd(
            "neml2.compile uses the in-process torch.compile / AOTAutograd path, "
            "not AOTI export; AOTAutograd's internal autograd is legitimate here"
        ):
            return compiled_forward(*args, **kwargs)

    # Instance-level override: nn.Module.__call__ dispatches to ``self.forward``.
    object.__setattr__(model, "forward", _forward)
    object.__setattr__(model, _COMPILED_MARK, True)
