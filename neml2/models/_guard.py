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

"""Runtime guard banning AOTI-incompatible / discouraged ``torch`` calls inside a
Python-native :class:`~neml2.model.Model` forward evaluation.

The Python-native authoring contract forbids a set of operations inside a
``forward`` / chain-rule body because they do not survive ``torch.export`` /
AOTInductor, or fuse poorly in Inductor (see ``DECISION.md`` D-051 / D-060 and
``RISK.md`` R-007 / R-011):

* **autograd / functional transforms** — ``torch.func.{vmap, jvp, vjp, jacrev,
  jacfwd, hessian, grad}``, ``torch.vmap``, ``torch.autograd.grad`` /
  ``backward``, ``torch.autograd.functional.*``, ``Tensor.backward``,
  ``torch.enable_grad``. The ``torch.func`` transforms / forward-mode dual never
  survive export; derive Jacobians analytically through the ``forward(v=...)``
  chain-rule machinery.
* **``torch.einsum``** — does not fuse cleanly with neighbouring pointwise ops in
  Inductor; use explicit ``matmul`` / broadcast-multiply-sum.

The ban is a default, not an absolute: **single reverse-mode** ``torch.autograd.grad``
*does* lower through AOTInductor (with ``torch._dynamo.config.trace_autograd_ops``;
proven by the parameter-derivative path). The framework uses it deliberately in two
sanctioned, framework-managed places that wrap the call in :func:`allow_autograd`:
the parameter derivatives (:mod:`neml2.models.param_ad`) and the ``request_AD``
auto-derived input chain rule (:mod:`neml2.models.input_ad`, see
:meth:`neml2.models.model.Model.request_AD`). The guard still fires for *hand-written*
autograd inside a leaf ``forward`` — that is the case it is meant to catch.

Mechanism: importing this module monkeypatches the offending callables *once,
process-wide*, with thin wrappers that delegate to the original implementation
**except** while a NEML2-native forward is on the stack — in which case they
raise a clear, actionable :class:`RuntimeError`. Outside a forward (``depth ==
0``) the wrappers are transparent, so ordinary ``torch`` use (training loops,
reference Jacobians in tests, etc.) is unaffected.

The forward window is opened/closed by :meth:`Model.__call__` via
:func:`_enter_forward` / :func:`_exit_forward` (re-entrant: nested child models
just bump the depth). The guard is therefore active during both eager evaluation
and ``torch.export`` tracing of a model.

Escape hatches: :func:`allow_autograd` and :func:`allow_einsum` are context
managers that temporarily lift a single category — each **requires a non-empty
reason** explaining why the otherwise-banned call is necessary.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager

import torch

__all__ = ["allow_autograd", "allow_einsum"]

# Category keys.
_AUTOGRAD = "autograd"
_EINSUM = "einsum"
#: A registered parameter read as a plain ``self.<attr>`` inside a forward. This
#: one is not a monkeypatch on a torch callable -- it is checked in
#: :meth:`Model.__getattr__` -- but it reuses the same depth + allow-counter
#: machinery so the forward window and an internal escape hatch behave uniformly.
_PARAM_ATTR = "param_attr"

# Per-thread state: forward-nesting depth + per-category allow-counters.
_state = threading.local()


def _depth() -> int:
    return getattr(_state, "depth", 0)


def _allow_counts() -> dict[str, int]:
    counts = getattr(_state, "allow", None)
    if counts is None:
        counts = {}
        _state.allow = counts
    return counts


def _enter_forward() -> None:
    """Open a forward window (called by ``Model.__call__``)."""
    _state.depth = _depth() + 1


def _exit_forward() -> None:
    """Close a forward window. Clamped at 0 so it stays balanced under errors."""
    _state.depth = max(0, _depth() - 1)


def _armed(category: str) -> bool:
    """True iff a forward is active and *category* is not currently allowed.

    Always False while ``torch.export`` / ``torch.compile`` is tracing. The guard
    is an EAGER authoring check -- it catches *hand-written* autograd / einsum in a
    ``forward`` -- and the framework's own sanctioned reverse-mode autodiff
    (parameter derivatives and ``request_AD``) must trace cleanly into the exported
    graph. Disabling the guard under compilation lets those graphs lower without the
    ``@contextmanager`` escape hatches (which Dynamo cannot trace through its block
    stack), at no loss of protection: an invalid model trips the guard the moment it
    runs eager, long before anyone exports it. ``is_compiling()`` is only consulted
    on the would-arm path (depth > 0, category not allowed), so eager has no extra
    cost on the common path.
    """
    return (
        _depth() > 0 and _allow_counts().get(category, 0) == 0 and not torch.compiler.is_compiling()
    )


_MESSAGES = {
    _AUTOGRAD: (
        "{name} is not compatible with torch.export / AOTInductor and is forbidden "
        "inside a NEML2-native Model.forward: higher-order autodiff transforms and "
        "autograd do not survive export (see DECISION.md D-051, RISK.md R-007/R-011). "
        "Derive the Jacobian analytically through the chain-rule machinery "
        "(`forward(v=...)`) instead. If you genuinely need this outside the export "
        "path, wrap the region in `neml2.allow_autograd(reason=...)`."
    ),
    _EINSUM: (
        "torch.einsum is disallowed inside a NEML2-native Model.forward: it does not "
        "fuse cleanly with neighbouring pointwise ops in Inductor (CLAUDE.md). Use "
        "explicit matmul (`@`) or broadcast-multiply-sum instead. If einsum is "
        "genuinely the clearest form for a contraction, wrap it in "
        "`neml2.allow_einsum(reason=...)`."
    ),
}


def _check(category: str, name: str) -> None:
    if _armed(category):
        raise RuntimeError(_MESSAGES[category].format(name=name))


# --------------------------------------------------------------------------- #
# Registered-parameter attribute-read guard
# --------------------------------------------------------------------------- #
def param_attr_guarded() -> bool:
    """True iff a NEML2-native forward is active and a registered-parameter
    attribute read is not currently permitted.

    Checked by :meth:`neml2.models.model.Model.__getattr__` to forbid reading a
    registered parameter as ``self.<attr>`` inside ``forward`` -- that bypasses
    the static-or-promoted dispatch in :meth:`Model._get_param` and silently
    breaks parameter promotion. The internal :func:`_allow_param_attr` window
    (used by ``_get_param`` itself) lifts it.
    """
    return _armed(_PARAM_ATTR)


@contextmanager
def _allow_param_attr() -> Generator[None, None, None]:
    """Internal escape hatch: permit registered-parameter attribute reads for the
    duration of the block. Used by :meth:`Model._get_param`, the one sanctioned
    reader, so its own ``getattr(self, name)`` is not rejected by the guard."""
    counts = _allow_counts()
    counts[_PARAM_ATTR] = counts.get(_PARAM_ATTR, 0) + 1
    try:
        yield
    finally:
        counts[_PARAM_ATTR] = max(0, counts.get(_PARAM_ATTR, 0) - 1)


# --------------------------------------------------------------------------- #
# Public escape-hatch context managers
# --------------------------------------------------------------------------- #
def _require_reason(reason: str) -> None:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            "A non-empty reason is required to lift a NEML2-native forward "
            "restriction — explain why the otherwise-banned call is necessary."
        )


@contextmanager
def _allow(category: str, reason: str) -> Generator[None, None, None]:
    _require_reason(reason)
    counts = _allow_counts()
    counts[category] = counts.get(category, 0) + 1
    try:
        yield
    finally:
        counts[category] = max(0, counts.get(category, 0) - 1)


def allow_autograd(reason: str) -> AbstractContextManager[None]:
    """Temporarily permit autograd / functional-transform calls inside a forward.

    Use only when the call genuinely runs *outside* the AOTI export path (e.g. an
    implicit-function-theorem adjoint in eager mode). ``reason`` must be a
    non-empty string explaining why.

    Usable as a context manager (``with allow_autograd("..."):``) or a decorator.
    """
    return _allow(_AUTOGRAD, reason)


def allow_einsum(reason: str) -> AbstractContextManager[None]:
    """Temporarily permit ``torch.einsum`` inside a forward.

    Prefer migrating to explicit ``matmul`` / broadcast-multiply-sum. ``reason``
    must be a non-empty string explaining why einsum is the clearest form here.

    Usable as a context manager (``with allow_einsum("..."):``) or a decorator.
    """
    return _allow(_EINSUM, reason)


# --------------------------------------------------------------------------- #
# One-time monkeypatch install
# --------------------------------------------------------------------------- #
_PATCHED = False


def _guarded(orig, category: str, name: str):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        _check(category, name)
        return orig(*args, **kwargs)

    wrapper.__neml2_guarded__ = True  # type: ignore[attr-defined]
    return wrapper


def _patch(obj, attr: str, category: str, public_name: str) -> None:
    if obj is None or not hasattr(obj, attr):
        return
    orig = getattr(obj, attr)
    if getattr(orig, "__neml2_guarded__", False):
        return  # already wrapped
    setattr(obj, attr, _guarded(orig, category, public_name))


def install() -> None:
    """Install the guards. Idempotent; called once on module import."""
    global _PATCHED
    if _PATCHED:
        return

    _patch(torch, "einsum", _EINSUM, "torch.einsum")

    # Functional transforms (torch.func.*) and torch.vmap.
    func = getattr(torch, "func", None)
    for name in ("vmap", "jvp", "vjp", "jacrev", "jacfwd", "hessian", "grad"):
        _patch(func, name, _AUTOGRAD, f"torch.func.{name}")
    _patch(torch, "vmap", _AUTOGRAD, "torch.vmap")

    # Autograd entry points.
    _patch(torch.autograd, "grad", _AUTOGRAD, "torch.autograd.grad")
    _patch(torch.autograd, "backward", _AUTOGRAD, "torch.autograd.backward")
    functional = getattr(torch.autograd, "functional", None)
    for name in ("jacobian", "hessian", "jvp", "vjp", "vhp", "hvp"):
        _patch(functional, name, _AUTOGRAD, f"torch.autograd.functional.{name}")
    _patch(torch.Tensor, "backward", _AUTOGRAD, "torch.Tensor.backward")
    _patch(torch, "enable_grad", _AUTOGRAD, "torch.enable_grad")

    _PATCHED = True


install()
