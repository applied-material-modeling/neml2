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
import re
from pathlib import Path
from typing import Union

SECTION_OPEN_RE = re.compile(r"^\[(.+?)\]$")
PARAM_RE = re.compile(r"^(\w[\w.]*)\s*=\s*(.+)$")


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


def _strip_value(raw: str) -> str:
    """Strip inline comments and surrounding single quotes from a parameter value."""
    value = raw.split("#")[0].strip()
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        value = value[1:-1]
    return value


def _node_to_entries(node: dict) -> list:
    """
    Convert an internal node's children into a list of ModelEntry objects.
    Each child of the node becomes one ModelEntry.
    """
    entries = []
    for child_name, child_nodes in node["children"].items():
        for child_node in child_nodes:
            params = dict(child_node["params"])
            type_val = params.get("type", "")
            # Recursively convert deeper nesting (rare, but handle it)
            nested_children = {}
            for grandchild_name, grandchild_nodes in child_node["children"].items():
                nested_children[grandchild_name] = _node_to_entries(
                    {"children": {grandchild_name: grandchild_nodes}}
                )
            entries.append(
                ModelEntry(
                    name=child_name,
                    type=type_val,
                    params=params,
                    children=nested_children,
                )
            )
    return entries


def parse_input(path: Union[str, Path]) -> dict:
    """
    Parse a HIT input file into a structured dictionary.

    The HIT format uses bracketed section headers ``[Name]`` to open a section
    and ``[]`` to close it. Sections can be nested. Key-value parameters inside
    a section follow the pattern ``key = value``.

    Args:
        path: Path to the HIT input file.

    Returns:
        A dict mapping top-level section names (e.g. ``"Models"``, ``"Tensors"``)
        to lists of :class:`ModelEntry` objects — one per named subsection.
        Multiple blocks with the same top-level section name (e.g. two
        ``[Models]`` blocks) are merged into a single list.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has unclosed sections or unmatched closing
            brackets.
    """
    path = Path(path)
    with open(path, "r") as f:
        lines = f.readlines()

    # Each node: {"params": {key: value, ...}, "children": {name: [node, ...]}}
    root_node: dict = {"params": {}, "children": {}}
    # Stack: list of (section_name, node_dict). Root has a dummy entry.
    stack = [("ROOT", root_node)]

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip().split("#")[0].strip()
        if not line:
            continue

        if line == "[]":
            if len(stack) == 1:
                raise ValueError(f"Unmatched [] at line {lineno}")
            stack.pop()
            continue

        m = SECTION_OPEN_RE.fullmatch(line)
        if m:
            sec_name = m.group(1).strip()
            parent_node = stack[-1][1]
            new_node: dict = {"params": {}, "children": {}}
            parent_node["children"].setdefault(sec_name, []).append(new_node)
            stack.append((sec_name, new_node))
            continue

        # Key-value parameter: capture only when inside a subsection (depth >= 2 means
        # stack has ROOT + top-level-section + current-entry = at least 3 entries)
        if len(stack) >= 3:
            pm = PARAM_RE.match(line)
            if pm:
                key = pm.group(1)
                value = _strip_value(pm.group(2))
                stack[-1][1]["params"][key] = value

    if len(stack) != 1:
        open_sections = [name for name, _ in stack[1:]]
        raise ValueError(f"Unclosed sections at EOF: {open_sections}")

    # Build result: for each top-level section, collect all entries across all
    # occurrences of that section (merging duplicate section names).
    result: dict = {}
    for sec_name, top_nodes in root_node["children"].items():
        entries: list = []
        for top_node in top_nodes:
            entries.extend(_node_to_entries(top_node))
        result[sec_name] = entries

    return result
