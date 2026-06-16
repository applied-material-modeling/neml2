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

"""``neml2-stub`` -- regenerate ``.pyi`` stubs for every pybind11 extension.

Walks the installed ``neml2`` package, finds every binary extension module
(currently just ``neml2.aoti._aoti``), and runs ``pybind11_stubgen`` against
each one. The generated stubs land next to their ``.so`` so pyright /
IDE autocompletion resolve ``from ._aoti import Model`` cleanly.

Invocation order matters: the script ``import``s every extension to
introspect it, so the package must be **fully installed** before this runs.
That means after ``pip install`` (CI runs ``neml2-stub`` between install
and pyright) or after ``cibuildwheel``'s repair step has unpacked the
wheel into the test env (``scripts/repair_wheel.py`` calls the helper
in-process, then re-packs the wheel with the stubs injected).

Any extra arguments are forwarded verbatim to ``pybind11_stubgen``
(e.g. ``neml2-stub --exit-code`` makes the script return non-zero when
stubgen reports unresolved names).
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import pkgutil
import sys
from pathlib import Path


def discover_extension_modules(package_name: str = "neml2") -> list[str]:
    """Return the dotted names of every pybind11 extension under ``package_name``.

    Uses :func:`pkgutil.walk_packages` to enumerate submodules and filters
    by spec origin -- a module is an extension iff its file extension is
    in :data:`importlib.machinery.EXTENSION_SUFFIXES` (e.g.
    ``.cpython-314-x86_64-linux-gnu.so`` on CPython 3.14 Linux).
    """
    package = importlib.import_module(package_name)
    prefix = f"{package_name}."
    found: list[str] = []
    for module in pkgutil.walk_packages(package.__path__, prefix):
        spec = importlib.util.find_spec(module.name)
        if spec is None or spec.origin is None:
            continue
        if any(spec.origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
            found.append(module.name)
    return sorted(found)


def _output_dir_for(module_name: str) -> Path:
    """Return the dir to pass to ``pybind11-stubgen -o`` for ``module_name``.

    pybind11-stubgen writes to ``<output_dir>/<dotted_path>.pyi``, so for
    the stub to land next to the ``.so`` we need ``output_dir`` =
    ``.so`` location minus the dotted-path's worth of directories.

    Computing this from the spec instead of ``package.__path__[0]`` matters
    for editable installs: ``neml2.__path__`` carries two entries (the
    minimal site-packages shim and the real source tree), and the shim's
    parent is site-packages -- writing the stub there strands it next to
    a different (incomplete) copy of ``neml2/`` instead of next to the
    extension. The spec's ``origin`` always points at the actual ``.so``,
    so this is robust to whatever the path-list ordering happens to be.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"can't resolve spec for {module_name}")
    depth = len(module_name.split("."))
    return Path(spec.origin).resolve().parents[depth - 1]


def generate_stubs(extra_args: list[str] | None = None) -> int:
    """Run ``pybind11_stubgen`` on every discovered extension.

    Each stub is written next to its ``.so`` (see
    :func:`_output_dir_for`). Returns the last non-zero exit code from
    ``pybind11_stubgen``, or 0 if every module succeeded. Missing
    ``pybind11_stubgen`` is a hard error -- this is a dev/CI tool and
    silent skipping would mask the pyright-fails-in-CI symptom that
    motivated it.
    """
    if importlib.util.find_spec("pybind11_stubgen") is None:
        print(
            "neml2-stub: pybind11-stubgen is not installed. Install it via "
            "`pip install pybind11-stubgen` (or the [dev] extras).",
            file=sys.stderr,
        )
        return 1

    import pybind11_stubgen  # noqa: PLC0415

    modules = discover_extension_modules("neml2")
    if not modules:
        print("neml2-stub: no pybind11 extension modules discovered under neml2/.")
        return 0

    last_failure = 0
    for module in modules:
        output_dir = _output_dir_for(module)
        argv = ["-o", str(output_dir), *(extra_args or []), module]
        print(f"neml2-stub: generating stubs for {module} -> {output_dir}")
        try:
            pybind11_stubgen.main(argv)
        except SystemExit as exc:
            if exc.code:
                print(
                    f"neml2-stub: pybind11-stubgen failed for {module} (exit {exc.code})",
                    file=sys.stderr,
                )
                last_failure = int(exc.code) if isinstance(exc.code, int) else 1
    return last_failure


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    return generate_stubs(args)


if __name__ == "__main__":
    raise SystemExit(main())
