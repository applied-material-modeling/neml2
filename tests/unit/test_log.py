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

"""Unit tests for the central verbosity/logging store (:mod:`neml2.log`).

Covers the ``NEML2_LOGS`` grammar, the env > programmatic > default precedence,
level/stream routing, and the configurable sink. The store is a process-global
singleton; the autouse fixture resets it around each test (pytest-xdist isolates
by process, so there is no cross-worker race)."""

from __future__ import annotations

import io

import pytest

from neml2 import log


@pytest.fixture(autouse=True)
def _clean_log(monkeypatch):
    """Start and end each test with a pristine store: no env, no programmatic
    defaults, the default stdout/stderr sink."""
    monkeypatch.delenv("NEML2_LOGS", raising=False)
    log.reload()
    log.reset_defaults()
    log.reset_sink()
    yield
    monkeypatch.delenv("NEML2_LOGS", raising=False)
    log.reload()
    log.reset_defaults()
    log.reset_sink()


def test_default_level_is_warning():
    for ch in log.CHANNELS:
        assert log.effective_level(ch) == "warning"
    assert log.enabled("model", "warning")
    assert not log.enabled("model", "info")
    assert not log.enabled("newton", "debug")


def test_programmatic_default_per_channel_and_all():
    log.set_default_level("newton", "debug")
    assert log.effective_level("newton") == "debug"
    assert log.effective_level("substep") == "warning"  # untouched
    log.set_default_level("all", "info")
    assert log.effective_level("substep") == "info"  # picks up the all baseline
    assert log.effective_level("newton") == "debug"  # explicit still wins over all
    log.reset_defaults()
    assert log.effective_level("newton") == "warning"


def test_env_channel_and_all(monkeypatch):
    monkeypatch.setenv("NEML2_LOGS", "all=info,newton=silent")
    log.reload()
    assert log.effective_level("newton") == "silent"  # explicit beats all
    assert log.effective_level("linear") == "info"
    assert log.effective_level("driver") == "info"


def test_env_beats_programmatic(monkeypatch):
    log.set_default_level("newton", "debug")  # a downstream app's default
    monkeypatch.setenv("NEML2_LOGS", "newton=silent")  # the end-user override
    log.reload()
    assert log.effective_level("newton") == "silent"
    # a channel the env does not name still falls back to the programmatic default
    log.set_default_level("substep", "info")
    assert log.effective_level("substep") == "info"


def test_level_integer_and_alias_synonyms(monkeypatch):
    monkeypatch.setenv("NEML2_LOGS", "newton=2,substep=3,model=off,driver=warn")
    log.reload()
    assert log.effective_level("newton") == "info"
    assert log.effective_level("substep") == "debug"
    assert log.effective_level("model") == "silent"
    assert log.effective_level("driver") == "warning"


def test_malformed_env_raises(monkeypatch):
    monkeypatch.setenv("NEML2_LOGS", "newton=bogus")
    with pytest.raises(ValueError, match="level"):
        log.reload()
    monkeypatch.setenv("NEML2_LOGS", "nosuchchannel=info")
    with pytest.raises(ValueError, match="channel"):
        log.reload()


def test_api_validates_channel_and_level():
    with pytest.raises(ValueError, match="level"):
        log.set_default_level("newton", "loud")
    with pytest.raises(ValueError, match="channel"):
        log.set_default_level("nope", "info")
    with pytest.raises(ValueError, match="channel"):
        log.enabled("nope", "info")


def test_emit_routes_info_stdout_warning_debug_stderr(capsys):
    log.set_default_level("all", "debug")
    log.emit("newton", "info", "an-info")
    log.emit("newton", "warning", "a-warning")
    log.emit("newton", "debug", "a-debug")
    out, err = capsys.readouterr()
    assert "an-info" in out and "[neml2:newton" in out
    assert "an-info" not in err
    assert "a-warning" in err
    assert "a-debug" in err


def test_emit_gated_by_level(capsys):
    log.set_default_level("newton", "info")
    log.emit("newton", "info", "shown")
    log.emit("newton", "debug", "hidden")
    out, err = capsys.readouterr()
    assert "shown" in out + err
    assert "hidden" not in out + err


def test_set_stream_redirects_all():
    buf = io.StringIO()
    log.set_default_level("all", "debug")
    log.set_stream(buf)
    log.emit("newton", "info", "x-info")
    log.emit("newton", "debug", "x-debug")
    s = buf.getvalue()
    assert "x-info" in s and "x-debug" in s
    assert "[neml2:newton" in s


def test_set_streams_splits_out_err():
    out, err = io.StringIO(), io.StringIO()
    log.set_default_level("all", "debug")
    log.set_streams(out, err)
    log.emit("newton", "info", "to-out")
    log.emit("newton", "debug", "to-err")
    assert "to-out" in out.getvalue() and "to-out" not in err.getvalue()
    assert "to-err" in err.getvalue() and "to-err" not in out.getvalue()


def test_set_sink_custom():
    events: list[tuple[str, str]] = []
    log.set_default_level("all", "debug")
    log.set_sink(lambda level, line: events.append((level, line)))
    log.emit("model", "warning", "hey")
    assert len(events) == 1
    level, line = events[0]
    assert level == "warning"
    assert line.startswith("[neml2:model") and line.endswith("] hey")


def test_set_sink_none_resets_to_default():
    log.set_sink(lambda level, line: None)
    log.set_sink(None)  # None restores the default stdout/stderr split
    assert log._user_sink is None


def test_pystore_fallback(monkeypatch):
    """The pure-Python fallback store (used when the ``_aoti`` backend is absent)
    mirrors the C++ store: same precedence, levels, and gating. Exercised directly
    since the backend is present in this build."""
    from neml2.log import _PyStore

    st = _PyStore()
    # Built-in default is warning.
    assert st.effective_level("newton") == "warning"
    assert st.enabled("model", "warning") and not st.enabled("model", "info")

    # Programmatic: per-channel, all baseline, explicit beats all, reset.
    st.set_default_level("newton", "debug")
    assert st.effective_level("newton") == "debug"
    st.set_default_level("all", "info")
    assert st.effective_level("substep") == "info"
    assert st.effective_level("newton") == "debug"
    st.reset_defaults()
    assert st.effective_level("newton") == "warning"

    # Env layer: all + channel override, and env beats programmatic.
    monkeypatch.setenv("NEML2_LOGS", "all=info,newton=silent")
    st.apply_env()
    assert st.effective_level("newton") == "silent"
    assert st.effective_level("linear") == "info"
    st.set_default_level("newton", "debug")
    assert st.effective_level("newton") == "silent"  # env still wins

    # Integer + alias synonyms.
    monkeypatch.setenv("NEML2_LOGS", "newton=2,model=off")
    st.apply_env()
    assert st.effective_level("newton") == "info"
    assert st.effective_level("model") == "silent"

    # Malformed spec raises.
    monkeypatch.setenv("NEML2_LOGS", "newton=bogus")
    with pytest.raises(ValueError):
        st.apply_env()

    # emit() routes through the shared forwarding sink (padded prefix), gated by level.
    events: list[tuple[str, str]] = []
    monkeypatch.delenv("NEML2_LOGS", raising=False)
    st.apply_env()
    st.set_default_level("all", "debug")
    st.set_default_level("linear", "silent")
    log.set_sink(lambda level, line: events.append((level, line)))
    try:
        st.emit("model", "debug", "shown")
        st.emit("linear", "debug", "hidden")  # linear is silent -> suppressed
    finally:
        log.reset_sink()
    lines = [ln for _, ln in events]
    assert any(ln.endswith("] shown") and ln.startswith("[neml2:model") for ln in lines)
    assert not any("hidden" in ln for ln in lines)
