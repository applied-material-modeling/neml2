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

"""``CSV<Type>`` user-tensor classes used by the verification suite.

Each class reads a CSV file (loaded lazily, cached per-path) and returns a
typed ``TensorWrapper`` whose leading axis indexes rows. The HIT options
mirror the C++ ``VTestTimeSeries`` API but read CSV rather than the
``.vtest`` text format (a one-time ``scripts/vtest_to_csv.py`` conversion
produced the CSVs).

Two column-selection modes:

* ``variable = 'foo'`` -- auto-builds column names by suffix (e.g. SR2
  => ``foo_xx`` ... ``foo_xy``). Used by the converted ``.vtest``-derived
  files where every variable follows the suffix convention.
* ``column_names = 'a b c'`` -- explicit column list. Used by hand-authored
  ``reference.csv`` files (e.g. the traction-separation scenarios).

SR2 storage: the ``.vtest`` files explicitly declare "Mandel convention"
in their headers, so ``CSVSR2`` returns the stacked columns verbatim with
no sqrt(2) scaling. If a future CSV stores physical (non-Mandel) shear values,
add an ``apply_mandel = true`` HIT option here at that point.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import nmhit
import torch

from ..factory import register_neml2_object
from ..schema import HitSchema, option
from ..types import SR2, WR2, Scalar, Vec

if TYPE_CHECKING:
    from ..factory import _NativeInputFile
    from ..types._base import TensorWrapper


@cache
def _load_csv(path: Path) -> dict[str, torch.Tensor]:
    """Read *path* once and return a dict of column-name -> 1-D float64 tensor.

    Uses the stdlib ``csv`` module to avoid pulling pandas into the import
    chain just for a few-hundred-row reference file.
    """
    import csv as _csv

    with path.open() as f:
        reader = _csv.reader(f)
        header = next(reader)
        rows = list(reader)

    cols: dict[str, list[float]] = {name: [] for name in header}
    for row in rows:
        if len(row) != len(header):
            raise ValueError(f"{path}: row has {len(row)} fields but header has {len(header)}")
        for name, value in zip(header, row, strict=True):
            cols[name].append(float(value))
    return {name: torch.tensor(vals, dtype=torch.float64) for name, vals in cols.items()}


def _resolve_csv_path(node: nmhit.Node, factory: _NativeInputFile) -> Path:
    raw = node.param_str("csv_file")
    path = Path(raw)
    if not path.is_absolute():
        path = factory._path.resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return path


def _read_optional_column_names(node: nmhit.Node) -> list[str] | None:
    """Read the optional ``column_names`` HIT param as a list of tokens."""
    raw = node.param_optional_str("column_names", "")
    if not raw:
        return None
    return raw.split()


def _select_columns(
    cols: dict[str, torch.Tensor],
    names: list[str],
    csv_path: Path,
) -> list[torch.Tensor]:
    missing = [n for n in names if n not in cols]
    if missing:
        raise KeyError(f"{csv_path}: missing column(s) {missing}; available: {sorted(cols)}")
    return [cols[n] for n in names]


class _CSVTensorBase:
    """Shared HIT-parsing scaffolding for the CSV<Type> classes.

    Subclasses set ``TYPE_NAME`` (registry key), ``WRAPPER`` (the typed
    ``TensorWrapper`` class), and override ``_default_column_names`` to
    expand a ``variable`` prefix into per-component column names.
    """

    TYPE_NAME: ClassVar[str]
    WRAPPER: ClassVar[type[TensorWrapper]]
    SECTION: ClassVar[str] = "Tensors"

    # Shared documentation schema — every CSV<Type> subclass inherits the same
    # HIT surface via :meth:`from_hit`. ``variable`` and ``column_names`` are
    # mutually exclusive (validated at parse time) but the schema can't express
    # that constraint; both are declared optional and the constructor enforces
    # the rule.
    hit = HitSchema(
        option(
            "csv_file",
            str,
            "Path to the CSV file. Resolved relative to the input file's directory when "
            "not absolute.",
        ),
        option(
            "variable",
            str,
            "Column-name prefix expanded via the per-type suffix convention (e.g. "
            "``stress`` -> ``stress_xx``, ``stress_yy``, ... for SR2). Mutually exclusive "
            "with ``column_names``.",
            default="",
        ),
        option(
            "column_names",
            list,
            "Explicit whitespace-separated list of CSV column names; bypasses the "
            "``variable`` suffix expansion. Mutually exclusive with ``variable``.",
            default=[],
        ),
    )

    @classmethod
    def _default_column_names(cls, variable: str) -> list[str]:
        raise NotImplementedError

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> TensorWrapper:
        csv_path = _resolve_csv_path(node, factory)
        cols = _load_csv(csv_path)

        explicit = _read_optional_column_names(node)
        variable = node.param_optional_str("variable", "")
        if explicit is not None and variable:
            raise ValueError(
                f"{cls.TYPE_NAME}: specify either 'variable' or 'column_names', not both "
                f"(at {csv_path})"
            )
        if explicit is not None:
            names = explicit
        elif variable:
            names = cls._default_column_names(variable)
        else:
            raise ValueError(
                f"{cls.TYPE_NAME}: must specify 'variable' or 'column_names' (at {csv_path})"
            )

        columns = _select_columns(cols, names, csv_path)
        return cls._build(columns)

    @classmethod
    def _build(cls, columns: list[torch.Tensor]) -> TensorWrapper:
        raise NotImplementedError


@register_neml2_object("CSVScalar")
class CSVScalar(_CSVTensorBase):
    """Load a single column from CSV as a ``Scalar`` with shape ``(N,)``."""

    TYPE_NAME = "CSVScalar"
    WRAPPER = Scalar

    @classmethod
    def _default_column_names(cls, variable: str) -> list[str]:
        return [variable]

    @classmethod
    def _build(cls, columns: list[torch.Tensor]) -> Scalar:
        if len(columns) != 1:
            raise ValueError(f"CSVScalar expects exactly 1 column, got {len(columns)}")
        return Scalar(columns[0])


@register_neml2_object("CSVSR2")
class CSVSR2(_CSVTensorBase):
    """Load 6 columns from CSV as an ``SR2`` with shape ``(N, 6)``.

    Column-name suffix order matches the C++ ``VTestTimeSeries<SR2>`` /
    Mandel slot convention: ``var_xx, var_yy, var_zz, var_yz, var_xz, var_xy``.
    Values are stacked verbatim -- ``.vtest`` files declare Mandel convention
    so the on-disk columns already carry the sqrt(2) scaling on shear slots.
    """

    TYPE_NAME = "CSVSR2"
    WRAPPER = SR2
    _SUFFIXES = ("xx", "yy", "zz", "yz", "xz", "xy")

    @classmethod
    def _default_column_names(cls, variable: str) -> list[str]:
        return [f"{variable}_{s}" for s in cls._SUFFIXES]

    @classmethod
    def _build(cls, columns: list[torch.Tensor]) -> SR2:
        if len(columns) != 6:
            raise ValueError(f"CSVSR2 expects exactly 6 columns, got {len(columns)}")
        return SR2(torch.stack(columns, dim=-1))


@register_neml2_object("CSVVec")
class CSVVec(_CSVTensorBase):
    """Load 3 columns from CSV as a ``Vec`` with shape ``(N, 3)``."""

    TYPE_NAME = "CSVVec"
    WRAPPER = Vec
    _SUFFIXES = ("x", "y", "z")

    @classmethod
    def _default_column_names(cls, variable: str) -> list[str]:
        return [f"{variable}_{s}" for s in cls._SUFFIXES]

    @classmethod
    def _build(cls, columns: list[torch.Tensor]) -> Vec:
        if len(columns) != 3:
            raise ValueError(f"CSVVec expects exactly 3 columns, got {len(columns)}")
        return Vec(torch.stack(columns, dim=-1))


@register_neml2_object("CSVWR2")
class CSVWR2(_CSVTensorBase):
    """Load 3 columns from CSV as a ``WR2`` (skew) with shape ``(N, 3)``.

    Column-name suffix order mirrors the C++ ``VTestTimeSeries<WR2>`` call:
    ``var_zy, var_xz, var_yx``. WR2 has no Mandel scaling -- values are
    stacked verbatim.
    """

    TYPE_NAME = "CSVWR2"
    WRAPPER = WR2
    _SUFFIXES = ("zy", "xz", "yx")

    @classmethod
    def _default_column_names(cls, variable: str) -> list[str]:
        return [f"{variable}_{s}" for s in cls._SUFFIXES]

    @classmethod
    def _build(cls, columns: list[torch.Tensor]) -> WR2:
        if len(columns) != 3:
            raise ValueError(f"CSVWR2 expects exactly 3 columns, got {len(columns)}")
        return WR2(torch.stack(columns, dim=-1))


__all__ = ["CSVScalar", "CSVSR2", "CSVVec", "CSVWR2"]
