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

"""Shared ``--load`` plumbing for the ``neml2-*`` CLI utilities.

Downstream users author custom ``Model`` / ``Driver`` / ``Solver`` / ``Data``
/ tensor classes outside the ``neml2`` package. Importing the module that
defines them is what fires the ``@register_native`` decorators that make the
types resolvable from a HIT input file.

A user-friendly CLI workflow needs a way to do that without writing a wrapper
script. :func:`add_load_argument` adds a cumulative ``--load`` flag that
accepts either:

* a path to a ``.py`` file or a package directory, or
* a dotted module name already importable from ``sys.path``.

:func:`load_user_extensions` then imports each path before the CLI's normal
load / run / inspect / compile workflow begins. The helper is invoked from
every CLI entry point so the user-facing surface stays uniform.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path


def add_load_argument(parser: argparse.ArgumentParser) -> None:
    """Add a cumulative ``--load PATH`` argument to *parser*.

    Each ``--load`` entry is imported before the CLI's normal workflow runs,
    so that the user-defined classes are registered in the native factory.
    """
    parser.add_argument(
        "--load",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Import a user extension before processing. PATH is either a "
            "filesystem path to a .py file / package directory, or a dotted "
            "module name already importable from sys.path. Repeatable; "
            "modules are imported in the order given so later modules may "
            "depend on names registered by earlier ones."
        ),
    )


def load_user_extensions(paths: list[str]) -> None:
    """Import every entry in *paths* so its ``@register_native`` decorators fire.

    File paths take precedence over dotted-module resolution: if *path* names
    an existing file or directory on disk, it is imported by spec under a
    sanitized synthetic module name; otherwise it falls through to
    :func:`importlib.import_module`.

    Raises :class:`ImportError` (with the offending path in the message) on
    the first failure — partial loads are not committed to the registry.
    """
    for path in paths:
        _import_one(path)


def _import_one(path: str) -> None:
    fs_target = Path(path)
    if fs_target.exists():
        _import_by_path(fs_target)
        return
    try:
        importlib.import_module(path)
    except ImportError as exc:
        raise ImportError(
            f"--load {path!r}: not a filesystem path and not an importable dotted module "
            f"({exc}). Pass either a .py file, a package directory, or a dotted module "
            "name on sys.path."
        ) from exc


def _import_by_path(target: Path) -> None:
    """Import a file or package directory by spec.

    Directories must contain ``__init__.py``. The synthetic module name is
    derived from the file/directory stem with non-identifier characters
    replaced — the name is only meaningful inside :data:`sys.modules`, so the
    sanitisation just keeps ``importlib`` happy.
    """
    resolved = target.resolve()
    if resolved.is_dir():
        init = resolved / "__init__.py"
        if not init.is_file():
            raise ImportError(
                f"--load {target!s}: directory has no __init__.py — point at the "
                "package's __init__.py directly or pass the parent dotted module name."
            )
        location = init
        # Add the directory's *parent* to sys.path so submodule imports inside
        # the package resolve against the user's expected layout.
        parent = str(resolved.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        module_name = _sanitise_module_name(resolved.name)
    else:
        location = resolved
        # Single-file scripts often `import` sibling modules — put the script's
        # directory on sys.path so those imports resolve the way the user
        # expects when running the script directly.
        parent = str(resolved.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        module_name = _sanitise_module_name(resolved.stem)

    spec = importlib.util.spec_from_file_location(module_name, location)
    if spec is None or spec.loader is None:
        raise ImportError(f"--load {target!s}: importlib could not build a module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Roll back the sys.modules entry so a broken extension doesn't
        # masquerade as successfully loaded on a retry.
        sys.modules.pop(module_name, None)
        raise


def _sanitise_module_name(stem: str) -> str:
    """Map an arbitrary file stem to a valid Python identifier.

    The result lives only in ``sys.modules`` under a ``_neml2_ext_`` prefix
    so it can never collide with the user's own module names.
    """
    safe = "".join(c if c.isidentifier() or c.isdigit() else "_" for c in stem)
    if not safe or not safe[0].isidentifier():
        safe = f"_{safe}"
    return f"_neml2_ext_{safe}"


__all__ = ["add_load_argument", "load_user_extensions"]
