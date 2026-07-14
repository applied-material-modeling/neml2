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

"""Central verbosity / logging control for NEML2 -- shared by all six routes.

Every neml2-emitted *library diagnostic* (solver traces, substepping, model /
equation-system debug, driver notices) is gated through one config store. The
same store lives in ``libneml2.so`` (so the C++ solvers obey it directly, whether
reached from C++ or from Python) and is surfaced here; this module is a thin
client plus the output sink.

Control it two ways, resolved highest-first:

1. the ``NEML2_LOGS`` environment variable (the end-user override) --
   ``TORCH_LOGS``-style, e.g. ``NEML2_LOGS="newton=info,substep=debug"`` or
   ``NEML2_LOGS="all=warning"``;
2. programmatic defaults set by a downstream app via :func:`set_default_level`.

Then the built-in default (``warning``) fills the rest. Levels, least to most
verbose: ``silent < warning < info < debug``. Channels: ``newton``, ``linear``,
``substep``, ``model``, ``tensor``, ``driver`` (plus ``all`` as a baseline).

Output goes to ``sys.stdout`` for ``info`` and ``sys.stderr`` for
``warning`` / ``debug`` by default; redirect with :func:`set_stream`,
:func:`set_streams`, or :func:`set_sink`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Protocol, cast

#: Level names, ordered least-to-most verbose (mirrors ``neml2/csrc/aoti/log.h``).
LEVELS = ("silent", "warning", "info", "debug")
#: Diagnostic channels (mirrors ``neml2/csrc/aoti/log.h``).
CHANNELS = ("newton", "linear", "substep", "model", "tensor", "driver")

# Canonical level constants for callers who prefer symbols over string literals.
SILENT = "silent"
WARNING = "warning"
INFO = "info"
DEBUG = "debug"

_LEVEL_ORD = {name: i for i, name in enumerate(LEVELS)}
# Accepted aliases -> canonical level (mirrors C++ parse_level).
_LEVEL_ALIASES = {
    "off": "silent",
    "none": "silent",
    "warn": "warning",
    "0": "silent",
    "1": "warning",
    "2": "info",
    "3": "debug",
}


def _norm_level(level: str) -> str:
    """Canonical level name, or raise ``ValueError`` on an unrecognized token."""
    key = str(level).strip().lower()
    key = _LEVEL_ALIASES.get(key, key)
    if key not in _LEVEL_ORD:
        raise ValueError(
            f"unknown log level {level!r} (expected one of {', '.join(LEVELS)} or 0..3)"
        )
    return key


def _norm_channel(channel: str, *, allow_all: bool = False) -> str:
    """Canonical channel name, or raise ``ValueError`` on an unrecognized token."""
    key = str(channel).strip().lower()
    if key == "all" and allow_all:
        return "all"
    if key not in CHANNELS:
        extra = "all, " if allow_all else ""
        raise ValueError(
            f"unknown log channel {channel!r} (expected one of {extra}{', '.join(CHANNELS)})"
        )
    return key


# ---------------------------------------------------------------------------
# Backend: the one store in libneml2.so, reached through the _aoti binding. When
# the compiled extension is unavailable (a partial build), fall back to a pure-
# Python store so Python-only diagnostics (and the AOTI-import-failure advisory)
# still function.
# ---------------------------------------------------------------------------
try:
    from neml2.aoti._aoti import log as _backend  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 (any import failure -> degrade to the Python store)
    _backend = None


class _PyStore:
    """Pure-Python mirror of the C++ store for partial builds (no ``_aoti``)."""

    def __init__(self) -> None:
        self._prog: dict[str, str] = {}
        self._prog_all: str | None = None
        self._env: dict[str, str] = {}
        self._env_all: str | None = None
        self._applied = False

    def apply_env(self) -> None:
        import os  # noqa: PLC0415

        self._env = {}
        self._env_all = None
        self._applied = True
        spec = os.environ.get("NEML2_LOGS")
        if not spec:
            return
        for entry in spec.split(","):
            if "=" not in entry:
                continue
            chan, _, lvl = entry.partition("=")
            level = _norm_level(lvl)
            chan = _norm_channel(chan, allow_all=True)
            if chan == "all":
                self._env_all = level
            else:
                self._env[chan] = level

    def effective_level(self, channel: str) -> str:
        if not self._applied:
            self.apply_env()
        if channel in self._env:
            return self._env[channel]
        if self._env_all is not None:
            return self._env_all
        if channel in self._prog:
            return self._prog[channel]
        if self._prog_all is not None:
            return self._prog_all
        return "warning"

    def enabled(self, channel: str, level: str) -> bool:
        return _LEVEL_ORD[self.effective_level(channel)] >= _LEVEL_ORD[level]

    def emit(self, channel: str, level: str, message: str) -> None:
        # Pad the channel to the longest name ("substep") for column alignment,
        # matching the C++ store's format().
        if self.enabled(channel, level):
            _forward(level, f"[neml2:{channel:<7}] {message}")

    def set_default_level(self, channel: str, level: str) -> None:
        if channel == "all":
            self._prog_all = level
        else:
            self._prog[channel] = level

    def reset_defaults(self) -> None:
        self._prog = {}
        self._prog_all = None


class _StoreProto(Protocol):
    def enabled(self, channel: str, level: str) -> bool: ...
    def effective_level(self, channel: str) -> str: ...
    def emit(self, channel: str, level: str, message: str) -> None: ...
    def set_default_level(self, channel: str, level: str) -> None: ...
    def reset_defaults(self) -> None: ...
    def apply_env(self) -> None: ...


#: The active store: the C++ one (via _aoti) when available, else the Python
#: fallback. Both expose the same method surface.
_store: _StoreProto = cast("_StoreProto", _backend if _backend is not None else _PyStore())


# ---------------------------------------------------------------------------
# Output sink. A single forwarding function is installed into the C++ store at
# import; it reads the module-level stream config at call time so redirection
# (notebooks, pytest capsys, set_stream) takes effect immediately.
# ---------------------------------------------------------------------------
_user_sink: Callable[[str, str], None] | None = None
_out = None  # None -> sys.stdout looked up dynamically
_err = None  # None -> sys.stderr looked up dynamically


def _forward(level: str, line: str) -> None:
    """The installed sink: route one already-formatted line by level."""
    if _user_sink is not None:
        _user_sink(level, line)
        return
    if level == "info":
        stream = _out if _out is not None else sys.stdout
    else:
        stream = _err if _err is not None else sys.stderr
    print(line, file=stream, flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def enabled(channel: str, level: str) -> bool:
    """Whether ``channel`` would emit at ``level`` (guard expensive log lines)."""
    channel = _norm_channel(channel)
    level = _norm_level(level)
    return bool(_store.enabled(channel, level))


def effective_level(channel: str) -> str:
    """The resolved level of ``channel`` (env > programmatic > built-in default)."""
    channel = _norm_channel(channel)
    return str(_store.effective_level(channel))


def emit(channel: str, level: str, message: str) -> None:
    """Emit ``message`` on ``channel`` at ``level`` iff enabled.

    The line is formatted as ``[neml2:<channel>] <message>`` and routed through
    the configured sink -- identical to a line the C++ solvers emit.
    """
    channel = _norm_channel(channel)
    level = _norm_level(level)
    _store.emit(channel, level, message)


def set_default_level(channel: str, level: str) -> None:
    """Set the programmatic default level for one channel (or ``"all"``).

    A downstream app calls this to pick its baseline verbosity; an ``NEML2_LOGS``
    entry for the same channel still overrides it.
    """
    channel = _norm_channel(channel, allow_all=True)
    level = _norm_level(level)
    _store.set_default_level(channel, level)


def reset_defaults() -> None:
    """Clear all programmatic defaults (revert to env > built-in default)."""
    _store.reset_defaults()


def reload() -> None:
    """Re-read ``NEML2_LOGS`` into the env layer (e.g. after changing it in a test)."""
    # Validate in Python first so a malformed spec raises a consistent ValueError
    # regardless of backend.
    import os  # noqa: PLC0415

    spec = os.environ.get("NEML2_LOGS")
    if spec:
        for entry in spec.split(","):
            if "=" not in entry:
                continue
            chan, _, lvl = entry.partition("=")
            _norm_level(lvl)
            _norm_channel(chan, allow_all=True)
    _store.apply_env()


def set_stream(stream) -> None:
    """Route *all* neml2 log output to a single stream (both info and warn/debug)."""
    global _out, _err, _user_sink
    _out = stream
    _err = stream
    _user_sink = None


def set_streams(out, err) -> None:
    """Route ``info`` to ``out`` and ``warning`` / ``debug`` to ``err``."""
    global _out, _err, _user_sink
    _out = out
    _err = err
    _user_sink = None


def set_sink(sink: Callable[[str, str], None] | None) -> None:
    """Install a custom sink ``sink(level_name, formatted_line)`` for all output.

    Pass ``None`` to restore the default stdout/stderr split.
    """
    global _user_sink
    if sink is None:
        reset_sink()
        return
    _user_sink = sink


def reset_sink() -> None:
    """Restore the default sink: ``info`` -> stdout, ``warning`` / ``debug`` -> stderr."""
    global _out, _err, _user_sink
    _out = None
    _err = None
    _user_sink = None


# Install the forwarding sink into the C++ store and prime the env layer so that
# all neml2 output -- including lines emitted from C++ -- flows through Python's
# streams (and is captured by notebooks / pytest).
if _backend is not None:
    _backend.set_sink(_forward)
reload()

__all__ = [
    "LEVELS",
    "CHANNELS",
    "SILENT",
    "WARNING",
    "INFO",
    "DEBUG",
    "enabled",
    "effective_level",
    "emit",
    "set_default_level",
    "reset_defaults",
    "reload",
    "set_stream",
    "set_streams",
    "set_sink",
    "reset_sink",
]
