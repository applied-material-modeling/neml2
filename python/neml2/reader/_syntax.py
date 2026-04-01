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

import dataclasses
import os
import re
import warnings
from pathlib import Path
from typing import Optional, Union

_LATEX_RE = re.compile(r"\\f\$(.*?)\\f\$", re.DOTALL)


def _strip_latex(text: str) -> str:
    """Remove Doxygen LaTeX delimiters (\\f$ ... \\f$) from a string."""
    if not text:
        return text
    return _LATEX_RE.sub(r"\1", text)


@dataclasses.dataclass
class ParamInfo:
    """Documentation for a single parameter of a NEML2 type."""

    name: str
    ftype: str
    """Field type: 'NONE', 'INPUT', 'OUTPUT', or 'PARAMETER'."""
    doc: str
    default: Optional[str]


@dataclasses.dataclass
class TypeInfo:
    """Documentation for a NEML2 type looked up from the syntax database."""

    qualified_name: str
    """Fully-qualified C++ name, e.g. 'neml2::LinearIsotropicElasticity'."""
    section: str
    """The input file section this type belongs to, e.g. 'Models'."""
    doc: str
    params: list
    """Non-suppressed parameters (excluding the 'type' entry itself)."""


class SyntaxDB:
    """
    Loads and queries the NEML2 syntax database (``syntax.yml``).

    The database maps C++ class names to their input-file documentation
    including parameter descriptions.

    Args:
        syntax_path: Explicit path to ``syntax.yml``. If ``None``, the
            database is auto-discovered by checking the ``NEML2_SYNTAX_DB``
            environment variable and then ``build/doc/syntax.yml`` relative
            to the repository root.
    """

    def __init__(self, syntax_path: Union[str, Path]):
        self._path: Path = Path(syntax_path)
        self._by_qualified: Optional[dict] = None
        self._by_short: Optional[dict] = None
        self._loaded = False

    @property
    def available(self) -> bool:
        """``True`` if the syntax database was found and can be loaded."""
        return self._path is not None and self._path.exists()

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        if not self.available:
            warnings.warn(
                "NEML2 syntax database (syntax.yml) not found. "
                "Type descriptions will be unavailable. "
                "Set the NEML2_SYNTAX_DB environment variable to the path of syntax.yml.",
                stacklevel=3,
            )
            self._by_qualified = {}
            self._by_short = {}
            return

        try:
            import yaml
        except ImportError as e:
            warnings.warn(
                f"PyYAML is required to load the syntax database: {e}. "
                "Install it with: pip install PyYAML",
                stacklevel=3,
            )
            self._by_qualified = {}
            self._by_short = {}
            return

        with open(self._path, "r") as f:
            raw = yaml.safe_load(f)

        self._by_qualified = {}
        self._by_short = {}

        for qualified_name, entry in raw.items():
            if not isinstance(entry, dict):
                continue

            section = entry.get("section", "")
            doc = _strip_latex(entry.get("doc") or "")

            params = []
            for param_name, param_data in entry.items():
                if param_name in ("section", "doc"):
                    continue
                if not isinstance(param_data, dict):
                    continue
                if param_name.startswith("_"):
                    continue
                if param_name == "type":
                    continue
                if param_data.get("suppressed", 0):
                    continue

                param_doc = _strip_latex(param_data.get("doc") or "")
                default_raw = param_data.get("value")
                default = (
                    str(default_raw) if default_raw is not None and default_raw != "" else None
                )

                params.append(
                    ParamInfo(
                        name=param_name,
                        ftype=param_data.get("ftype", "NONE"),
                        doc=param_doc,
                        default=default,
                    )
                )

            info = TypeInfo(
                qualified_name=qualified_name,
                section=section,
                doc=doc,
                params=params,
            )
            self._by_qualified[qualified_name] = info

            # Short name: everything after the last '::'
            short = qualified_name.rsplit("::", 1)[-1]
            # Only store short name if it's unique; if a collision exists, keep first
            if short not in self._by_short:
                self._by_short[short] = info

    def lookup(self, type_name: str) -> Optional[TypeInfo]:
        """
        Look up a type by its short name or fully-qualified name.

        Args:
            type_name: e.g. ``"LinearIsotropicElasticity"`` or
                ``"neml2::LinearIsotropicElasticity"``.

        Returns:
            :class:`TypeInfo` if found, ``None`` otherwise.
        """
        self._ensure_loaded()

        if "::" in type_name:
            assert self._by_qualified is not None
            return self._by_qualified.get(type_name)

        assert self._by_short is not None
        return self._by_short.get(type_name)
