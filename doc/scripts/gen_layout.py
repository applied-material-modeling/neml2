#!/usr/bin/env python

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

"""Expand the source ``DoxygenLayout.xml`` template into the two Doxygen
layout files actually consumed by the build:

* ``build/doc/DoxygenLayout.xml``         — for the C++ Doxygen run
* ``build/doc/DoxygenLayoutPython.xml``   — for the Python Doxygen run

The source template contains a ``<syntax-trees/>`` placeholder which is
expanded into the per-section / per-submodule navigation hierarchy
derived from ``syntax.json`` (the output of ``neml2-syntax --json``).
The Python flavor is then derived from the populated C++ flavor by
rewriting URL forms and swapping the C++/Python API-Reference subtrees.
"""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from syntax_to_md import (
    all_submodule_nodes,
    build_tree,
    direct_children,
    display_name,
    submodule_anchor,
    type_anchor,
)

# Order the syntax sections appear in the sidebar. Sections not listed
# here are appended afterwards in alphabetical order.
SECTION_ORDER = [
    "Settings",
    "Tensors",
    "Models",
    "EquationSystems",
    "Solvers",
    "Data",
    "Drivers",
    "Schedulers",
]


def _section_title(section: str) -> str:
    return f"[{section}]"


def _submodule_label(parts: tuple[str, ...]) -> str:
    return display_name(parts[-1])


def _make_tab(tab_type: str, url: str | None, title: str | None) -> ET.Element:
    attrs = {"type": tab_type}
    if url is not None:
        attrs["url"] = url
    if title is not None:
        attrs["title"] = title
    return ET.Element("tab", attrs)


def _render_node(
    section: str,
    parts: tuple[str, ...],
    groups: dict,
    nodes: list,
) -> ET.Element:
    """Render one navigation tab (with descendants) for a submodule node."""
    children_subs = direct_children(parts, nodes)
    children_types = groups.get(parts, [])
    title = _submodule_label(parts) if parts else _section_title(section)
    anchor = submodule_anchor(section, parts)
    has_children = bool(children_subs) or bool(children_types)
    tab = _make_tab("usergroup" if has_children else "user", f"@ref {anchor}", title)
    for child in children_subs:
        tab.append(_render_node(section, child, groups, nodes))
    for r in children_types:
        tab.append(_make_tab("user", f"@ref {type_anchor(r['type'])}", r["type"]))
    return tab


def render_syntax_subtree(records: list[dict]) -> list[ET.Element]:
    """Build the ordered list of top-level <tab> nodes for every syntax
    section. Returned tabs are inserted in place of ``<syntax-trees/>``.
    """
    tree = build_tree(records)
    ordered: list[str] = [s for s in SECTION_ORDER if s in tree]
    ordered += sorted(s for s in tree if s not in SECTION_ORDER)
    out: list[ET.Element] = []
    for section in ordered:
        groups = tree[section]
        nodes = all_submodule_nodes(groups)
        out.append(_render_node(section, (), groups, nodes))
    return out


def _replace_syntax_marker(root: ET.Element, replacements: list[ET.Element]) -> None:
    """Find the unique ``<syntax-trees/>`` placeholder and substitute it with
    the rendered section tabs in-place.
    """
    for parent in root.iter():
        for idx, child in enumerate(list(parent)):
            if child.tag == "syntax-trees":
                parent.remove(child)
                for offset, new_child in enumerate(replacements):
                    parent.insert(idx + offset, new_child)
                return
    raise RuntimeError("DoxygenLayout.xml does not contain a <syntax-trees/> placeholder")


def _ref_to_html_url(ref: str) -> str:
    # The Python doxygen build lives at html/python/ while the C++ build
    # is at html/. Every reference from the Python sidebar therefore
    # needs to point one level up and target the C++ HTML output. The
    # filenames Doxygen generates match the page anchors with a .html
    # suffix.
    return f"../{ref}.html"


def _python_transform_urls(root: ET.Element) -> None:
    for tab in root.iter("tab"):
        url = tab.get("url")
        if url and url.startswith("@ref "):
            tab.set("url", _ref_to_html_url(url[len("@ref ") :].strip()))


# In the source layout the C++ tree uses namespacelist/classlist tabs and
# the Python tree uses explicit user URLs into python/. In the Python
# flavor we want the reverse: C++ tree should point at user URLs that
# reach back into the C++ html output, and Python tree should use the
# Doxygen-builtin namespacelist/classlist tabs (which the Python doxygen
# run populates).
_API_TREE_TITLES = {
    "C++ API Reference": [
        ("user", "../namespaces.html", "Namespaces"),
        ("user", "../annotated.html", "Classes"),
    ],
    "Python API Reference": [
        ("namespacelist", None, "Namespaces"),
        ("classlist", None, "Classes"),
    ],
}


def _python_swap_api_trees(root: ET.Element) -> None:
    for parent in root.iter("tab"):
        title = parent.get("title")
        if title not in _API_TREE_TITLES:
            continue
        for child in list(parent):
            parent.remove(child)
        for tab_type, url, child_title in _API_TREE_TITLES[title]:
            child = ET.SubElement(parent, "tab", {"type": tab_type})
            if url is not None:
                child.set("url", url)
            if child_title is not None:
                child.set("title", child_title)
            if tab_type in {"namespacelist", "classlist"}:
                child.set("visible", "yes")
                child.set("intro", "")


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + ("  " * level)
    if len(elem) > 0:
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for i, child in enumerate(elem):
            _indent(child, level + 1)
            tail = pad + "  " if i + 1 < len(elem) else pad
            if not child.tail or not child.tail.strip():
                child.tail = tail
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def _write_layout(root: ET.Element, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    ET.ElementTree(root).write(dest, encoding="unicode", xml_declaration=False)
    # ET.write doesn't add a trailing newline.
    with open(dest, "a") as f:
        f.write("\n")


def generate_layouts(source_layout: Path, syntax_json: Path, build_dir: Path) -> tuple[Path, Path]:
    """Read the source layout + syntax JSON, write the two generated layouts,
    and return their paths.
    """
    records = json.loads(syntax_json.read_text())
    replacements = render_syntax_subtree(records)

    # C++ flavor: source layout + syntax tree substitution.
    cpp_root = ET.parse(source_layout).getroot()
    _replace_syntax_marker(cpp_root, replacements)
    cpp_dest = build_dir / "DoxygenLayout.xml"
    _write_layout(cpp_root, cpp_dest)

    # Python flavor: deep copy of the populated C++ tree, then URL rewrite
    # and API-Reference subtree swap.
    py_root = copy.deepcopy(cpp_root)
    _python_transform_urls(py_root)
    _python_swap_api_trees(py_root)
    py_dest = build_dir / "DoxygenLayoutPython.xml"
    _write_layout(py_root, py_dest)

    return cpp_dest, py_dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate DoxygenLayout.xml variants.")
    parser.add_argument("--source", required=True, type=Path, help="Source DoxygenLayout.xml")
    parser.add_argument("--syntax", required=True, type=Path, help="syntax.json")
    parser.add_argument("--build-dir", required=True, type=Path, help="Build directory")
    args = parser.parse_args()
    cpp, py = generate_layouts(args.source, args.syntax, args.build_dir)
    print(f"wrote {cpp}")
    print(f"wrote {py}")
