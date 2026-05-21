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

"""Convert the JSON syntax dump emitted by ``neml2-syntax --json`` into
one markdown page per registered object, grouped by the source folder
hierarchy. The :func:`build_tree` helper is also imported by
``gen_layout.py`` to render the matching Doxygen navigation subtree.
"""

import json
import re
import shutil
import sys
import unicodedata as ud
from collections import defaultdict
from pathlib import Path

from loguru import logger

# Folder names that should not be auto-titlecased — uppercase acronyms, etc.
DISPLAY_NAME_OVERRIDES = {
    "kwn": "KWN",
}


def postprocess(value, type_str):
    if type_str == "bool":
        value = "true" if value else "false"
    return value


def ftype_icon(ftype):
    return {
        "INPUT": "🇮",
        "OUTPUT": "🇴",
        "PARAMETER": "🇵",
        "BUFFER": "🇧",
    }.get(ftype, "")


# Default character budget for the per-object blurb shown on index pages.
# Long enough to convey the gist; short enough that ~30 entries still fit
# on screen. Tweak in one place if reviewers want a different feel.
INDEX_BLURB_MAX_LEN = 120


# Doxygen math: \f$...\f$ (inline), \f[...\f] (display), \f{env}...\f} (env).
# `re.DOTALL` so multi-line math blocks are matched in one go.
_MATH_BLOCK_RE = re.compile(
    r"\\f\$.*?\\f\$|\\f\[.*?\\f\]|\\f\{[^}]*\}.*?\\f\}",
    re.DOTALL,
)


def truncate_doc(doc: str, max_len: int = INDEX_BLURB_MAX_LEN) -> str:
    """Collapse whitespace and truncate a multiline doc string to a single
    line suitable for the per-object blurb on index pages. Truncation
    happens at the last word boundary before ``max_len`` characters and is
    suffixed with " …". Empty input returns an empty string.

    Doxygen formula blocks (``\\f$ … \\f$``, ``\\f[ … \\f]``,
    ``\\f{env} … \\f}``) are replaced with ``[…]`` before truncation.
    Inside a markdown list item, Doxygen mis-parses formula contents (for
    example, ``\\dot{s}`` is read as the start of an inline ``\\dot``
    graph block) which produces a stream of "unknown command" warnings.
    The full math is preserved on the per-type detail page that the
    bullet links to.
    """
    if not doc:
        return ""
    stripped = _MATH_BLOCK_RE.sub("[…]", doc)
    flat = " ".join(stripped.split())
    if len(flat) <= max_len:
        return flat
    head = flat[:max_len].rsplit(" ", 1)[0]
    if not head:
        return ""
    return head + " …"


def display_name(folder: str) -> str:
    """Human-readable label for a folder segment."""
    if folder in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[folder]
    return folder.replace("_", " ").title()


def submodule_parts(source_path: str) -> tuple[str, ...]:
    """Extract the submodule path components from a `source_path` field.

    ``source_path`` looks like ``models/solid_mechanics/elasticity/Foo.cxx``
    or ``solvers/Newton.cxx`` or ``""`` (for Settings, which isn't
    registered through the C++ Registry).

    The first component (the section's source root, e.g. ``models``) and
    the trailing filename are stripped. Empty string -> ``()``.
    """
    if not source_path:
        return ()
    parts = source_path.split("/")
    if len(parts) <= 2:
        return ()
    return tuple(parts[1:-1])


def section_slug(section: str) -> str:
    return section.lower()


def submodule_anchor(section: str, parts: tuple[str, ...]) -> str:
    base = f"syntax-{section_slug(section)}"
    if not parts:
        return base
    return base + "-" + "-".join(p.replace("_", "-") for p in parts)


def type_anchor(type_name: str) -> str:
    # Preserved unchanged from the original generator so existing
    # ``[Foo](#foo)`` cross-references in tutorials keep resolving.
    return type_name.lower()


def submodule_title(parts: tuple[str, ...]) -> str:
    return " / ".join(display_name(p) for p in parts)


def section_prologue(section):
    # Reserved for future per-section context. The legend explaining the
    # 🇮/🇴/🇵/🇧 ftype icons used in Model options lives at the top of each
    # per-type Model page (see write_type_page); it's not pedagogically
    # useful on the section index which doesn't render those icons.
    del section
    return ""


def first_nonprintable(path: Path, encoding="utf-8") -> dict | None:
    BAD_CATEGORIES = {"Cc", "Cf", "Cs"}
    ALLOW = {"\n", "\r", "\t"}
    text = path.read_text(encoding=encoding, errors="surrogateescape")
    line = 1
    col = 1
    for idx, ch in enumerate(text):
        if ch == "\n":
            line += 1
            col = 1
            continue
        cat = ud.category(ch)
        if ch not in ALLOW and cat in BAD_CATEGORIES:
            return {
                "line": line,
                "col": col,
                "index": idx,
                "char": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "category": cat,
                "name": ud.name(ch, "<no name>"),
            }
        col += 1
    return None


def build_tree(records: list[dict]) -> dict:
    """Group records as {section: {submodule_parts: [record, ...]}}.

    Records within each leaf list are sorted by type name. The inner dict
    is plain (not defaultdict) and its keys are the literal parts tuples
    that appear in the data — submodule nodes with no direct child types
    (intermediate-only nodes) are added by ``all_submodule_nodes``.
    """
    tree: dict[str, dict[tuple[str, ...], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if not r.get("section"):
            continue
        tree[r["section"]][submodule_parts(r["source_path"])].append(r)
    out: dict[str, dict[tuple[str, ...], list[dict]]] = {}
    for section, groups in tree.items():
        sorted_groups: dict[tuple[str, ...], list[dict]] = {}
        for parts in sorted(groups):
            sorted_groups[parts] = sorted(groups[parts], key=lambda r: r["type"].lower())
        out[section] = sorted_groups
    return out


def all_submodule_nodes(
    groups: dict[tuple[str, ...], list[dict]],
) -> list[tuple[str, ...]]:
    """All submodule nodes that appear in a section, including ancestors.

    If only ``solid_mechanics/elasticity`` is present in ``groups``, the
    ancestor ``solid_mechanics`` is added too so its index page gets
    written even when no types are registered directly under it.
    """
    nodes: set[tuple[str, ...]] = set()
    for parts in groups:
        for i in range(1, len(parts) + 1):
            nodes.add(parts[:i])
    return sorted(nodes)


def direct_children(parent: tuple[str, ...], nodes: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return [n for n in nodes if len(n) == len(parent) + 1 and n[: len(parent)] == parent]


def section_page_path(outdir: Path, section: str) -> Path:
    return outdir / f"{section_slug(section)}.md"


def submodule_page_path(outdir: Path, section: str, parts: tuple[str, ...]) -> Path:
    return outdir / section_slug(section) / Path(*parts[:-1]) / f"{parts[-1]}.md"


def type_page_path(outdir: Path, section: str, parts: tuple[str, ...], type_name: str) -> Path:
    return outdir / section_slug(section) / Path(*parts) / f"{type_name}.md"


def write_option_block(stream, opt: dict, log, section: str, type_name: str) -> int:
    """Render one option as a markdown bullet with its metadata as sub-bullets.

    LaTeX (``\\f$ ... \\f$``) is allowed in option descriptions — Doxygen
    processes math directly in markdown lists. The older ``<details>`` /
    ``<summary>`` HTML form silently stripped escaped backslashes inside
    the summary, which is why the legacy generator banned LaTeX there.
    """
    missing = 0
    name = opt["name"]
    icon = ftype_icon(opt["ftype"])
    icon_part = f" {icon}" if icon else ""
    if not opt["doc"]:
        stream.write(f"- `{name}`{icon_part}\n")
        missing += 1
        log.write(f"  * '{section}/{type_name}/{name}' is missing option description\n")
    else:
        stream.write(f"- `{name}`{icon_part} — {opt['doc']}\n")
    stream.write(f"  - **Type**: {opt['type']}\n")
    stream.write("  - **Required**: {}\n".format("Yes" if opt["required"] else "No"))
    value = postprocess(opt["value"], opt["type"])
    if value:
        stream.write(f"  - **Default**: {value}\n")
    return missing


def write_type_page(
    outdir: Path,
    section: str,
    parts: tuple[str, ...],
    record: dict,
    log,
) -> int:
    missing = 0
    type_name = record["type"]
    path = type_page_path(outdir, section, parts, type_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = record.get("options", [])
    has_ftype_icons = any(opt.get("ftype") not in (None, "", "NONE") for opt in options)
    with open(path, "w") as stream:
        stream.write(f"# {type_name} {{#{type_anchor(type_name)}}}\n\n")
        if record["doc"]:
            stream.write(record["doc"] + "\n\n")
        else:
            missing += 1
            log.write(f"  * '{section}/{type_name}' is missing object description\n")
        if has_ftype_icons:
            stream.write(
                "\\note\n**Option roles**: 🇮 input · 🇴 output · 🇵 parameter · 🇧 buffer\n\n"
            )
        for opt in options:
            missing += write_option_block(stream, opt, log, section, type_name)
        stream.write("\n")
        class_name = record.get("class_name", "")
        if class_name:
            stream.write(f"Source documentation [link](@ref {class_name})\n")
    return missing


def write_submodule_index(
    outdir: Path,
    section: str,
    parts: tuple[str, ...],
    groups: dict[tuple[str, ...], list[dict]],
    nodes: list[tuple[str, ...]],
) -> None:
    path = submodule_page_path(outdir, section, parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        title = submodule_title(parts)
        stream.write(f"# [{section}] {title} {{#{submodule_anchor(section, parts)}}}\n\n")
        children = direct_children(parts, nodes)
        if children:
            stream.write("## Subgroups\n\n")
            for child in children:
                stream.write(
                    f"- [{display_name(child[-1])}](@ref {submodule_anchor(section, child)})\n"
                )
            stream.write("\n")
        types = groups.get(parts, [])
        if types:
            stream.write("## Objects\n\n")
            for r in types:
                blurb = truncate_doc(r.get("doc", ""))
                suffix = f" — {blurb}" if blurb else ""
                stream.write(f"- [{r['type']}](@ref {type_anchor(r['type'])}){suffix}\n")
            stream.write("\n")


def write_section_index(
    outdir: Path,
    section: str,
    groups: dict[tuple[str, ...], list[dict]],
    nodes: list[tuple[str, ...]],
) -> None:
    path = section_page_path(outdir, section)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        stream.write(f"# [{section}] {{#{submodule_anchor(section, ())}}}\n\n")
        stream.write(section_prologue(section))
        stream.write("\n")
        stream.write(
            f"Refer to [System Documentation](@ref system-{section_slug(section)})"
            " for detailed explanation about this system.\n\n"
        )
        children = direct_children((), nodes)
        if children:
            stream.write("## Subgroups\n\n")
            for child in children:
                stream.write(
                    f"- [{display_name(child[-1])}](@ref {submodule_anchor(section, child)})\n"
                )
            stream.write("\n")
        root_types = groups.get((), [])
        if root_types:
            stream.write("## Objects\n\n")
            for r in root_types:
                blurb = truncate_doc(r.get("doc", ""))
                suffix = f" — {blurb}" if blurb else ""
                stream.write(f"- [{r['type']}](@ref {type_anchor(r['type'])}){suffix}\n")
            stream.write("\n")


def syntax_to_md(syntax_file: Path, outdir: Path, logfile: Path) -> int:
    """Convert the JSON file emitted by ``neml2-syntax --json`` into a
    tree of markdown files mirroring the source folder hierarchy.

    Returns the count of missing documentation items.
    """
    with open(syntax_file) as stream:
        try:
            records = json.load(stream)
        except json.JSONDecodeError as e:
            logger.error(f"Error reading JSON file: {syntax_file}")
            logger.error(str(e))
            np = first_nonprintable(syntax_file)
            if np is not None:
                logger.error(
                    "The first non-printable character is at line {line}, "
                    "column {col} ({codepoint}, {category}, {name})".format(**np)
                )
            sys.exit(1)

    # Wipe the output directory first. Without this, a previous run's pages
    # for objects whose source files have since moved would survive and
    # produce Doxygen "duplicate id" warnings (every per-type page anchors
    # on the lowercased type name, so two copies of the same type from
    # different paths collide).
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)

    tree = build_tree(records)
    missing = 0

    with open(logfile, "w") as log:
        log.write("### Syntax check\n\n")
        for r in records:
            if not r.get("section"):
                missing += 1
                log.write(
                    "Section is not defined for one of the objects, did you forget to "
                    "set options.section() in one of its base classes?"
                )
        for section, groups in sorted(tree.items()):
            nodes = all_submodule_nodes(groups)
            write_section_index(outdir, section, groups, nodes)
            for parts in nodes:
                write_submodule_index(outdir, section, parts, groups, nodes)
            for parts, types in groups.items():
                for r in types:
                    missing += write_type_page(outdir, section, parts, r, log)

        if missing == 0:
            log.write("No syntax error, good job! :purple_heart:")
        else:
            print("*" * 79, file=sys.stderr)
            print("Syntax errors have been written to", logfile, file=sys.stderr)
            print("*" * 79, file=sys.stderr)

    return missing
