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
from pathlib import Path

import nmhit


@dataclasses.dataclass
class ModelEntry:
    """A single named subsection in a HIT input file."""

    name: str
    """The user-assigned name, e.g. 'isoharden'."""

    type: str
    """The value of the 'type' key, e.g. 'VoceIsotropicHardening'."""

    params: dict
    """All key-value pairs from the section body (including 'type')."""

    children: dict
    """Nested subsections as {section_name: [ModelEntry, ...]}."""


def _section_to_entry(section: nmhit.Section) -> ModelEntry:
    params = {f.path(): f.param_str() for f in section.children(nmhit.NodeType.Field)}
    children = {}
    for subsection in section.children(nmhit.NodeType.Section):
        sub_name = subsection.path()
        children[sub_name] = [
            _section_to_entry(s) for s in subsection.children(nmhit.NodeType.Section)
        ]
    return ModelEntry(
        name=section.path(),
        type=params.get("type", ""),
        params=params,
        children=children,
    )


def parse_input(path: str | Path) -> dict:
    """
    Parse a HIT input file into a structured dictionary.

    Args:
        path: Path to the HIT input file.

    Returns:
        A dict mapping top-level section names (e.g. ``"Models"``, ``"Tensors"``)
        to lists of :class:`ModelEntry` objects — one per named subsection.
        Multiple blocks with the same top-level section name are merged into a
        single list.

    Raises:
        FileNotFoundError: If the file does not exist.
        nmhit.Error: If the file contains invalid HIT syntax.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    root = nmhit.parse_file(path)
    result: dict = {}
    for top_section in root.children(nmhit.NodeType.Section):
        sec_name = top_section.path()
        result.setdefault(sec_name, []).extend(
            _section_to_entry(s) for s in top_section.children(nmhit.NodeType.Section)
        )
    return result
