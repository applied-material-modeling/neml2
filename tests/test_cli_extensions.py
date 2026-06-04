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

"""Tests for the shared ``--load`` extension loader."""

from __future__ import annotations

import io
import json
import sys
import textwrap
from pathlib import Path

import pytest

from neml2.cli import _extensions  # noqa: PLC2701
from neml2.cli import syntax as _syntax_cli
from neml2.factory import _registry

_EXT_BODY = textwrap.dedent(
    '''
    """Minimal user extension for the --load CLI tests."""

    from neml2.factory import register_native
    from neml2.schema import HitSchema, option


    @register_native("{type_name}")
    class {class_name}:
        """A custom user-defined Tensors class."""

        SECTION = "Tensors"

        hit = HitSchema(
            option("dummy", str, "Placeholder option."),
        )

        @classmethod
        def from_hit(cls, node, factory):
            del node, factory
            return cls()
    '''
)


def _write_extension(dir_path: Path, filename: str, type_name: str, class_name: str) -> Path:
    target = dir_path / filename
    target.write_text(_EXT_BODY.format(type_name=type_name, class_name=class_name))
    return target


@pytest.fixture
def cleanup_registry():
    """Remove any test-registered type names from the global registry on teardown.

    Module-level side effects (``@register_native``) are not reversible by
    Python alone — this fixture undoes them so registry-mutation tests don't
    pollute downstream tests.
    """
    snapshot = set(_registry.keys())
    yield
    for added in set(_registry.keys()) - snapshot:
        _registry.pop(added, None)


def test_load_extension_by_file_path_registers_native_type(tmp_path: Path, cleanup_registry: None):
    """A user .py file passed to ``--load`` makes its registered class visible
    in :data:`neml2.factory._registry` after :func:`load_user_extensions`."""
    _write_extension(tmp_path, "ext.py", "TestExtTypeFile", "TestExtFileClass")

    assert "TestExtTypeFile" not in _registry
    _extensions.load_user_extensions([str(tmp_path / "ext.py")])
    assert "TestExtTypeFile" in _registry


def test_load_extension_by_package_directory(tmp_path: Path, cleanup_registry: None):
    """Pointing ``--load`` at a directory imports its ``__init__.py``."""
    pkg = tmp_path / "ext_pkg"
    pkg.mkdir()
    _write_extension(pkg, "__init__.py", "TestExtTypePkg", "TestExtPkgClass")

    _extensions.load_user_extensions([str(pkg)])
    assert "TestExtTypePkg" in _registry


def test_load_extension_by_dotted_module_name(tmp_path: Path, cleanup_registry: None):
    """A dotted module on ``sys.path`` is imported via :func:`importlib.import_module`."""
    pkg = tmp_path / "ext_dotted_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    _write_extension(pkg, "sub.py", "TestExtTypeDotted", "TestExtDottedClass")
    sys.path.insert(0, str(tmp_path))
    try:
        _extensions.load_user_extensions(["ext_dotted_pkg.sub"])
    finally:
        sys.path.remove(str(tmp_path))
        # Drop the imported submodules so a re-import next test goes through
        # the spec loader fresh — repeated runs in the same pytest session
        # otherwise see a cached module that already registered its types.
        for name in [n for n in sys.modules if n.startswith("ext_dotted_pkg")]:
            sys.modules.pop(name, None)
    assert "TestExtTypeDotted" in _registry


def test_load_extension_bad_path_raises_import_error_with_clear_message(tmp_path: Path):
    """A non-existent path is reported through :class:`ImportError` with a
    message that points at the user-supplied value, so the CLI's outer
    error-printing surfaces what was tried."""
    with pytest.raises(ImportError, match="not a filesystem path and not an importable"):
        _extensions.load_user_extensions([str(tmp_path / "does_not_exist.py")])


def test_load_extension_directory_without_init_raises(tmp_path: Path):
    bare = tmp_path / "bare_dir"
    bare.mkdir()
    with pytest.raises(ImportError, match="no __init__.py"):
        _extensions.load_user_extensions([str(bare)])


def test_load_extension_ordering_is_preserved(tmp_path: Path, cleanup_registry: None):
    """Modules are imported in argument order — verifying so users can rely on
    one extension depending on names registered by an earlier one."""
    _write_extension(tmp_path, "first.py", "TestExtOrderFirst", "TestExtOrderFirstClass")
    second = tmp_path / "second.py"
    second.write_text(
        textwrap.dedent(
            '''
            """Depends on the first extension being already imported."""
            from neml2.factory import _registry

            assert "TestExtOrderFirst" in _registry, "second.py imported before first.py"
            '''
        )
    )
    _extensions.load_user_extensions([str(tmp_path / "first.py"), str(second)])
    # Reaching here means the assertion in second.py held — i.e. ordering was preserved.
    assert "TestExtOrderFirst" in _registry


def test_load_extension_rolls_back_sys_modules_on_failure(tmp_path: Path):
    """A broken extension must not leave a half-loaded entry in ``sys.modules``,
    otherwise a retry sees a successful no-op import."""
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(RuntimeError, match="boom"):
        _extensions.load_user_extensions([str(broken)])
    # No leftover module in sys.modules from the failed load.
    assert not any(name.endswith("broken") for name in sys.modules if "_neml2_ext_" in name)


def test_neml2_syntax_cli_load_surfaces_user_type(tmp_path: Path, cleanup_registry: None):
    """End-to-end smoke: ``neml2-syntax --load ... --type Foo --json -``
    returns the user-defined type's record."""
    _write_extension(tmp_path, "ext.py", "TestExtEndToEnd", "TestExtEndToEndClass")
    stdout = io.StringIO()
    rc = _syntax_cli.main(
        ["--load", str(tmp_path / "ext.py"), "--json", "-", "--type", "TestExtEndToEnd"],
        stdout=stdout,
    )
    assert rc == 0
    payload = json.loads(stdout.getvalue())
    assert len(payload) == 1
    assert payload[0]["type"] == "TestExtEndToEnd"
    assert payload[0]["section"] == "Tensors"
    assert [opt["name"] for opt in payload[0]["options"]] == ["dummy"]


def test_neml2_syntax_cli_reports_bad_load(tmp_path: Path):
    err = io.StringIO()
    rc = _syntax_cli.main(
        ["--load", str(tmp_path / "missing.py"), "--json", "-"],
        stdout=io.StringIO(),
        stderr=err,
    )
    assert rc == 1
    assert "missing.py" in err.getvalue()
