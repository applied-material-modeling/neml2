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
and writes one MyST `.md` per registered type under
``doc/generated/syntax/<section>/<TypeName>.md`` plus per-section + top-
level index pages. The output tree is gitignored.

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


def _render_section_index(section: str, types: Iterable[str]) -> str:
    titles = {
        "Models": "Models",
        "Tensors": "Tensors",
        "Solvers": "Solvers",
        "EquationSystems": "Equation systems",
        "Drivers": "Drivers",
        "Data": "Data",
    }
    title = titles.get(section, section)
    lines = [
        f"({section.lower()}-syntax)=",
        f"# {title}",
        "",
        f"All registered `[{section}]` types, autogenerated from `neml2-syntax --json`.",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        "",
    ]
    for t in sorted(types):
        lines.append(t)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _render_top_index(sections_with_types: dict[str, list[str]]) -> str:
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
    for section in SECTION_ORDER:
        if sections_with_types.get(section):
            lines.append(f"{section}/index")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _generate(srcdir: Path) -> None:
    entries = _run_neml2_syntax()

    by_section: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        section = entry.get("section") or ""
        if not section:
            # Entries without a section (e.g. the internal AOTIModel shim)
            # are intentionally hidden from the catalog.
            continue
        by_section[section].append(entry)

    gen_root = srcdir.joinpath(*GEN_SUBDIR)
    # Wipe stale per-type pages before regenerating so renames don't leave
    # orphans. We rebuild the whole tree, but use _write_if_changed below
    # so unchanged content doesn't bump file mtimes.
    if gen_root.exists():
        shutil.rmtree(gen_root)
    gen_root.mkdir(parents=True, exist_ok=True)

    written = 0
    section_type_map: dict[str, list[str]] = {}
    for section, items in by_section.items():
        section_dir = gen_root / section
        type_names: list[str] = []
        for entry in items:
            page = _render_type_page(entry)
            if _write_if_changed(section_dir / f"{entry['type']}.md", page):
                written += 1
            type_names.append(entry["type"])
        if _write_if_changed(section_dir / "index.md", _render_section_index(section, type_names)):
            written += 1
        section_type_map[section] = type_names

    if _write_if_changed(gen_root / "index.md", _render_top_index(section_type_map)):
        written += 1

    logger.info(
        "neml2_syntax: wrote %d/%d pages",
        written,
        sum(len(v) for v in by_section.values()) + len(by_section) + 1,
    )


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
