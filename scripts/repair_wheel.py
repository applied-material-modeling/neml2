#!/usr/bin/env python3
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

"""
Post-build wheel repair invoked by cibuildwheel's repair-wheel-command:

  Linux:  run auditwheel excluding PyTorch .so files (they are declared runtime
          deps and must not be bundled).
  macOS:  pass-through (delocate is skipped; torch dylibs are not bundleable and
          plain macosx_* tags are accepted by PyPI).

Both flows then install the repaired wheel into the current Python env,
run ``neml2-stub`` to generate ``.pyi`` files alongside the freshly-installed
extensions, and inject those stubs back into the wheel so the published
artifact ships them. End users get pyright-resolvable stubs without
having to run anything extra.

Usage: python repair_wheel.py {wheel} {dest_dir}
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _torch_excludes() -> list[str]:
    """Return --exclude flags for every .so in PyTorch's lib directory."""
    try:
        import torch

        lib_dir = Path(torch.__file__).parent / "lib"
        return [
            flag for f in lib_dir.iterdir() if ".so" in f.name for flag in ("--exclude", f.name)
        ]
    except ImportError:
        return []


def _install_and_generate_stubs(wheel: Path) -> tuple[Path, list[Path]]:
    """Install ``wheel`` into the current env, run ``neml2-stub``, return
    the installed ``neml2/`` dir + every ``.pyi`` it produced (recursively).

    Install order matters and is deliberate:

    1. The wheel itself goes in with ``--force-reinstall --no-deps`` --
       we want the freshly-built artifact to clobber any older install,
       but ``--force-reinstall`` cascading to torch would pointlessly
       re-download multiple GB.
    2. The wheel's *runtime* dependencies (read from its own
       ``importlib.metadata`` record, with extras filtered out) get
       installed in a second pass. pip's default
       ``upgrade-strategy=only-if-needed`` skips torch (already there
       from ``before-build``) and installs the rest -- currently
       ``nmhit`` and ``pyzag``. Doing it this way means new entries in
       ``[project.dependencies]`` in pyproject.toml propagate
       automatically; this script never needs editing.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
        check=True,
    )
    # Resolve runtime deps from the just-installed wheel's metadata.
    # importlib.metadata.distribution(...).requires returns strings like
    # ``"torch>=2.10.0"`` or ``"pytest; extra == 'dev'"``; we want only
    # the unconditional ones (no ``; extra ==`` marker), which are
    # exactly the runtime deps.
    import importlib.metadata  # noqa: PLC0415

    raw = importlib.metadata.distribution("neml2").requires or []
    runtime_deps = [spec for spec in raw if "extra" not in spec.split(";", 1)[-1].lower()]
    if runtime_deps:
        print(f"repair_wheel: installing runtime deps {runtime_deps}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *runtime_deps],
            check=True,
        )

    from neml2.cli.stub import generate_stubs

    rc = generate_stubs()
    if rc:
        raise RuntimeError(f"neml2-stub failed with exit code {rc}")

    import neml2  # noqa: PLC0415

    pkg_dir = Path(neml2.__path__[0])
    return pkg_dir, sorted(pkg_dir.rglob("*.pyi"))


def _inject_stubs(wheel: Path, pkg_dir: Path, stubs: list[Path], dest_dir: Path) -> None:
    """Unpack ``wheel``, drop each ``.pyi`` into its mirrored path under
    ``neml2/``, repack into ``dest_dir``.

    Stubs nested under subpackages (e.g. ``neml2/aoti/_aoti.pyi``) preserve
    their relative path from ``pkg_dir`` so the wheel layout exactly mirrors
    the installed layout the stub was written for.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["wheel", "unpack", str(wheel), "--dest", tmp], check=True)
        unpacked = next(Path(tmp).iterdir())
        for pyi in stubs:
            target = unpacked / "neml2" / pyi.relative_to(pkg_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(pyi, target)
            print(f"repair_wheel: injected {target.relative_to(unpacked)}")
        subprocess.run(
            ["wheel", "pack", str(unpacked), "--dest-dir", str(dest_dir)],
            check=True,
        )


def main() -> None:
    wheel = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])
    dest_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform.startswith("linux"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["auditwheel", "repair", "-w", tmp] + _torch_excludes() + [str(wheel)],
                check=True,
            )
            repaired = next(Path(tmp).glob("*.whl"))
            pkg_dir, stubs = _install_and_generate_stubs(repaired)
            _inject_stubs(repaired, pkg_dir, stubs, dest_dir)
    else:
        # macOS: pass-through; delocate would try to bundle torch dylibs.
        pkg_dir, stubs = _install_and_generate_stubs(wheel)
        _inject_stubs(wheel, pkg_dir, stubs, dest_dir)


if __name__ == "__main__":
    main()
