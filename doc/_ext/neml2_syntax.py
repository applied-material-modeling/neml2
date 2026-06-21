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

"""Sphinx extension: autogenerate the NEML2 syntax catalog.

Runs `neml2-syntax --json` once per build (at the `builder-inited` event)
and writes one MyST `.md` per registered type, nested by the type's
location in the source tree (from each record's ``source_path``):
``doc/generated/syntax/<Section>/<submodule>/.../<TypeName>.md``, plus an
index page at every level (top, section, and each submodule) so the
toctree mirrors the package layout. The output tree is gitignored.

The generator is incremental: it diffs the new content against any
existing file before writing, so unchanged pages don't trigger sphinx
rebuilds on `--watch` runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Pages live under <conf-dir>/generated/syntax/. They're regenerated on
# every build; the directory is in doc/.gitignore.
GEN_SUBDIR = ("generated", "syntax")

# Sections in the order they appear in the top-level index.
SECTION_ORDER = (
    "Models",
    "Tensors",
    "Solvers",
    "EquationSystems",
    "Drivers",
    "Data",
)


def _run_neml2_syntax() -> list[dict]:
    """Invoke `neml2-syntax --json -` and parse the result.

    Uses a temp file rather than stdout because the CLI's `--json -`
    path is not guaranteed and the writer prints status to stdout that
    would corrupt a pipe. Returns the raw list-of-dicts the CLI emits.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("r+", suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    try:
        result = subprocess.run(
            ["neml2-syntax", "--json", str(tmp)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"`neml2-syntax --json` failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        return json.loads(tmp.read_text())
    finally:
        tmp.unlink(missing_ok=True)


def _write_if_changed(path: Path, content: str) -> bool:
    """Write `content` to `path` iff different. Returns True if written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == content:
        return False
    path.write_text(content)
    return True


def _render_option_row(opt: dict) -> str:
    """Render one option as a definition-list entry (MyST deflist)."""
    name = opt.get("name", "?")
    otype = opt.get("type", "")
    ftype = opt.get("ftype", "NONE")
    required = opt.get("required", False)
    default = opt.get("value", "")
    doc = (opt.get("doc") or "").strip().replace("\n", " ")

    badges = []
    if ftype and ftype != "NONE":
        badges.append(f"`{ftype.lower()}`")
    if otype:
        badges.append(f"`{otype}`")
    if required:
        badges.append("**required**")
    if not required and default:
        badges.append(f"default `{default}`")

    badge_str = " &middot; ".join(badges)
    head = f"`{name}`"
    if badge_str:
        head += f" &mdash; {badge_str}"
    body = doc if doc else "*(no description)*"
    return f"{head}\n: {body}\n"


def _render_type_page(entry: dict) -> str:
    """Render the full per-type MyST page."""
    type_name = entry["type"]
    section = entry.get("section", "")
    source = entry.get("source_path", "")
    doc = (entry.get("doc") or "").strip()
    options = entry.get("options", []) or []

    label = f"({section.lower()}-{type_name})=" if section else f"({type_name})="
    lines = [
        label,
        f"# {type_name}",
        "",
    ]
    if source:
        lines.append(f"**Source:** `{source}`")
        lines.append("")
    if doc:
        lines.append(doc)
        lines.append("")

    # Group options by ftype so inputs/outputs/parameters are visually
    # separated — readers usually want one of these groups, not all.
    by_ftype: dict[str, list[dict]] = defaultdict(list)
    for opt in options:
        by_ftype[opt.get("ftype", "NONE")].append(opt)

    group_order = ("INPUT", "OUTPUT", "PARAMETER", "NONE")
    group_titles = {
        "INPUT": "Inputs",
        "OUTPUT": "Outputs",
        "PARAMETER": "Parameters",
        "NONE": "Other options",
    }
    for ftype in group_order:
        bucket = by_ftype.get(ftype)
        if not bucket:
            continue
        lines.append(f"## {group_titles[ftype]}")
        lines.append("")
        for opt in bucket:
            lines.append(_render_option_row(opt))
        lines.append("")

    if not options:
        lines.append("*This type has no documented options.*")
        lines.append("")

    return "\n".join(lines)


# Display titles for each section's index page.
SECTION_TITLES = {
    "Models": "Models",
    "Tensors": "Tensors",
    "Solvers": "Solvers",
    "EquationSystems": "Equation systems",
    "Drivers": "Drivers",
    "Data": "Data",
}


def _submodule_parts(source_path: str) -> tuple[str, ...]:
    """Submodule components of a registration site, mirroring the source tree.

    ``source_path`` looks like ``models/solid_mechanics/elasticity/Foo.py``.
    The leading section root (``models``) and the trailing filename are
    stripped, leaving the intermediate package path the catalog nests under.
    ``solvers/Newton.py`` -> ``()`` (a section-root type).
    """
    if not source_path:
        return ()
    parts = source_path.split("/")
    if len(parts) <= 2:
        return ()
    return tuple(parts[1:-1])


def _all_submodule_nodes(groups: dict[tuple[str, ...], list[dict]]) -> set[tuple[str, ...]]:
    """Every submodule node in a section, including ancestors with no direct types."""
    nodes: set[tuple[str, ...]] = set()
    for parts in groups:
        for i in range(1, len(parts) + 1):
            nodes.add(parts[:i])
    return nodes


def _child_nodes(parent: tuple[str, ...], nodes: set[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Direct children of ``parent`` among ``nodes`` (one level deeper)."""
    return sorted(n for n in nodes if len(n) == len(parent) + 1 and n[: len(parent)] == parent)


def _toctree(child_submodules: list[str], type_names: list[str]) -> list[str]:
    """A maxdepth-1 toctree: child submodule indexes, then sibling type pages."""
    lines = ["```{toctree}", ":maxdepth: 1", ""]
    lines += [f"{sub}/index" for sub in sorted(child_submodules)]
    lines += sorted(type_names)
    lines += ["```", ""]
    return lines


def _render_section_index(section: str, child_submodules: list[str], root_types: list[str]) -> str:
    title = SECTION_TITLES.get(section, section)
    lines = [
        f"({section.lower()}-syntax)=",
        f"# {title}",
        "",
        f"All registered `[{section}]` types, grouped by their location in the "
        "source tree and autogenerated from `neml2-syntax --json`.",
        "",
    ]
    lines += _toctree(child_submodules, root_types)
    return "\n".join(lines)


def _render_submodule_index(
    section: str, parts: tuple[str, ...], child_submodules: list[str], type_names: list[str]
) -> str:
    path_str = "/".join(parts)
    lines = [
        f"# {parts[-1]}",
        "",
        f"`[{section}]` types under `{path_str}/`.",
        "",
    ]
    lines += _toctree(child_submodules, type_names)
    return "\n".join(lines)


def _render_top_index(present_sections: Iterable[str]) -> str:
    present = set(present_sections)
    lines = [
        "(syntax-catalog)=",
        "# Syntax catalog",
        "",
        "Reference pages for every type registered with the NEML2 native "
        "factory, autogenerated by the `neml2_syntax` Sphinx extension on "
        "every build.",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        "",
    ]
    lines += [f"{section}/index" for section in SECTION_ORDER if section in present]
    lines += ["```", ""]
    return "\n".join(lines)


def _generate(srcdir: Path) -> None:
    entries = _run_neml2_syntax()

    # section -> submodule parts -> [entries], mirroring the source tree.
    by_section: dict[str, dict[tuple[str, ...], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        section = entry.get("section") or ""
        if not section:
            # Entries without a section (e.g. the internal AOTIModel shim)
            # are intentionally hidden from the catalog.
            continue
        parts = _submodule_parts(entry.get("source_path", ""))
        by_section[section][parts].append(entry)

    gen_root = srcdir.joinpath(*GEN_SUBDIR)
    # Wipe stale pages before regenerating so renames/moves don't leave
    # orphans. _write_if_changed below keeps unchanged content from bumping
    # mtimes on --watch rebuilds.
    if gen_root.exists():
        shutil.rmtree(gen_root)
    gen_root.mkdir(parents=True, exist_ok=True)

    written = 0
    total = 0
    for section, groups in by_section.items():
        section_dir = gen_root / section
        nodes = _all_submodule_nodes(groups)

        # Per-type pages, nested under their submodule path. The page label
        # stays `(<section>-<Type>)=` regardless of depth, so existing
        # `[](<section>-<Type>)` cross-references keep resolving after a move.
        for parts, items in groups.items():
            for entry in items:
                path = section_dir.joinpath(*parts) / f"{entry['type']}.md"
                if _write_if_changed(path, _render_type_page(entry)):
                    written += 1
                total += 1

        # Per-submodule index pages (with ancestors that hold no direct types).
        for node in sorted(nodes):
            child_subs = [c[-1] for c in _child_nodes(node, nodes)]
            type_names = [e["type"] for e in groups.get(node, [])]
            path = section_dir.joinpath(*node) / "index.md"
            page = _render_submodule_index(section, node, child_subs, type_names)
            if _write_if_changed(path, page):
                written += 1
            total += 1

        # Section index (the submodule-tree root).
        root_subs = [c[-1] for c in _child_nodes((), nodes)]
        root_types = [e["type"] for e in groups.get((), [])]
        page = _render_section_index(section, root_subs, root_types)
        if _write_if_changed(section_dir / "index.md", page):
            written += 1
        total += 1

    if _write_if_changed(gen_root / "index.md", _render_top_index(by_section.keys())):
        written += 1
    total += 1

    logger.info("neml2_syntax: wrote %d/%d pages", written, total)


def _builder_inited(app: Sphinx) -> None:
    try:
        _generate(Path(app.srcdir))
    except FileNotFoundError as e:
        logger.warning("neml2_syntax: `neml2-syntax` not on PATH; skipping catalog. (%s)", e)
    except Exception as e:  # noqa: BLE001 - we want any failure to be non-fatal
        logger.warning("neml2_syntax: catalog generation failed: %s", e)


def setup(app: Sphinx) -> dict:
    app.connect("builder-inited", _builder_inited)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
