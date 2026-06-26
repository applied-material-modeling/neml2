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

"""Manage the torch / NEML2 wheel compatibility matrix.

Usage:
  compat_matrix.py check
        verify compatibility.yaml internal consistency and sync with
        dependencies.yaml

  compat_matrix.py expand --kind {test|wheel|cibuildwheel} [--mode {full|pr}]
        print the matrix as JSON for use in a GitHub Actions strategy.matrix
        (test/wheel emit {"include": [...]}; cibuildwheel emits a flat list of
        cp tags, e.g. ["cp310","cp314"]; --mode pr emits the lean torch x
        python extremes subset for PRs)

  compat_matrix.py render [--in-place FILE] [--check]
        render the matrix as a Markdown table; with --in-place, splice it into
        FILE between BEGIN_COMPAT_MATRIX / END_COMPAT_MATRIX sentinels; with
        --check, exit 1 if the in-place rendering would change the file
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
COMPAT_FILE = REPO_ROOT / "doc/content/installation/compatibility.yaml"
DEPS_FILE = Path(__file__).parent / "dependencies.yaml"

# Sentinels for `render --in-place`. The note inside the BEGIN line tells humans
# not to hand-edit the table (the pre-commit hook will overwrite it).
RENDER_BEGIN = (
    "<!-- BEGIN_COMPAT_MATRIX -- "
    "regenerate with: python scripts/compat_matrix.py render --in-place FILE -->"
)
RENDER_END = "<!-- END_COMPAT_MATRIX -->"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code, text):
    return f"{code}{text}{RESET}" if _supports_color() else text


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_compat() -> dict:
    if not COMPAT_FILE.exists():
        return {"build_torch": "", "combinations": []}
    with open(COMPAT_FILE) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("combinations", [])
    return data


def load_deps() -> dict:
    with open(DEPS_FILE) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def _ver_tuple(v: str) -> tuple:
    try:
        return tuple(int(p) for p in v.split(".")[:3])
    except ValueError:
        return (0, 0, 0)


def cmd_check(_args) -> None:
    """Validate the hand-maintained matrix against the ranges in dependencies.yaml.

    build_torch == torch.version is enforced separately by ``dep_manager check``
    (build_torch carries a ``# dependencies: torch.version`` annotation), so we
    don't re-check it here. What's left is what only this script can see: every
    row sits within the advertised torch / python ranges, and there are no
    duplicate rows.
    """
    errors: list[str] = []
    data = load_compat()

    deps = load_deps()
    t_min = deps["torch"].get("version_min")
    t_max = deps["torch"].get("version_max")
    py_min = deps["python"].get("version_min")

    for i, row in enumerate(data["combinations"]):
        loc = f"combinations[{i}]"
        for key in ("torch", "python", "os"):
            if key not in row:
                errors.append(f"{loc}: missing '{key}'")
        # Bounds: every row must sit within the advertised ranges so the matrix,
        # the pyproject.toml constraint, and the doc page can't drift.
        t = row.get("torch")
        if t is not None and t_min is not None and _ver_tuple(t) < _ver_tuple(t_min):
            errors.append(f"{loc}: torch={t!r} < torch.version_min={t_min!r}")
        if t is not None and t_max is not None and _ver_tuple(t) > _ver_tuple(t_max):
            errors.append(f"{loc}: torch={t!r} > torch.version_max={t_max!r}")
        p = row.get("python")
        if p is not None and py_min is not None and _ver_tuple(p) < _ver_tuple(py_min):
            errors.append(f"{loc}: python={p!r} < python.version_min={py_min!r}")

    # Reject duplicate rows -- they'd just cost CI minutes for no signal.
    keys = [(r.get("torch"), r.get("python"), r.get("os")) for r in data["combinations"]]
    seen = set()
    for k in keys:
        if k in seen:
            errors.append(f"duplicate row: {k}")
        seen.add(k)

    # Coverage: the advertised boundary versions must actually be exercised by a
    # row, else we'd claim support for a min/max we never test.
    pythons = {r.get("python") for r in data["combinations"]}
    torches = {r.get("torch") for r in data["combinations"]}
    if py_min is not None and py_min not in pythons:
        errors.append(f"python.version_min={py_min!r} is advertised but no row tests it")
    if t_min is not None and t_min not in torches:
        errors.append(f"torch.version_min={t_min!r} is advertised but no row tests it")
    if t_max is not None and t_max not in torches:
        errors.append(f"torch.version_max={t_max!r} is advertised but no row tests it")

    if errors:
        print(_c(RED, "FAIL") + f": {len(errors)} issue(s) in {COMPAT_FILE.name}:")
        for e in errors:
            print(f"  {_c(RED, '✗')} {e}")
        sys.exit(1)
    print(
        _c(GREEN, "OK") + f": {len(data['combinations'])} rows, "
        f"torch range=[{t_min}, {t_max}], python>={py_min}"
    )


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------


def _subset(combinations: list[dict]) -> list[dict]:
    """The PR-time subset: torch and python *extremes* only, all OS kept.

    Keeps rows whose torch is the min or max torch AND whose python is the min
    or max python present in the matrix -- the four corners per OS, where
    breakage is most likely. The full matrix (run on push to main) covers the
    interior, so PRs get a fast, low-noise signal without losing the boundary
    coverage.
    """
    torches = sorted({r["torch"] for r in combinations}, key=_ver_tuple)
    pys = sorted({r["python"] for r in combinations}, key=_ver_tuple)
    keep_t = {torches[0], torches[-1]}
    keep_p = {pys[0], pys[-1]}
    return [r for r in combinations if r["torch"] in keep_t and r["python"] in keep_p]


def cmd_expand(args) -> None:
    data = load_compat()
    combinations = data["combinations"]
    if args.mode == "pr":
        combinations = _subset(combinations)
    if args.kind == "cibuildwheel":
        # Distinct python versions in cibuildwheel build-identifier form
        # (3.10 -> cp310), as a flat JSON list for a GitHub Actions matrix axis.
        pys = sorted({r["python"] for r in combinations}, key=_ver_tuple)
        print(json.dumps(["cp" + p.replace(".", "") for p in pys], separators=(",", ":")))
        return
    if args.kind == "test":
        # Default channel: every row installs a CPU-index torch at runtime.
        include = [
            {"torch": r["torch"], "python": r["python"], "os": r["os"], "channel": "cpu"}
            for r in combinations
        ]
        # cuda-runtime rows: the wheel is built against CPU torch, so additionally
        # import + cpu-test it under a *cuda* torch to catch any ABI / DT_NEEDED
        # skew between the two PyTorch distribution channels -- the bug class that
        # made the AOTI binding fail to import under a cpu-only torch. cuda torch
        # ships only for Linux, and this is a C++-ABI check that does not vary
        # with the Python version, so cover just the Linux torch extremes at the
        # newest Python. Full mode only (push to main / workflow_dispatch): the
        # cuda wheel is ~2.5 GB to install, so PRs skip it for a fast signal.
        if args.mode == "full":
            linux = [r for r in data["combinations"] if r["os"] == "ubuntu-latest"]
            if linux:
                torches = sorted({r["torch"] for r in linux}, key=_ver_tuple)
                newest_py = sorted({r["python"] for r in linux}, key=_ver_tuple)[-1]
                for t in sorted({torches[0], torches[-1]}, key=_ver_tuple):
                    include.append(
                        {"torch": t, "python": newest_py, "os": "ubuntu-latest", "channel": "cuda"}
                    )
    elif args.kind == "wheel":
        seen: set[tuple[str, str]] = set()
        include = []
        for r in combinations:
            key = (r["python"], r["os"])
            if key in seen:
                continue
            seen.add(key)
            include.append({"python": r["python"], "os": r["os"]})
    else:
        print(f"Unknown --kind: {args.kind}", file=sys.stderr)
        sys.exit(1)
    # GitHub Actions expects a single-line string when used with fromJson().
    print(json.dumps({"include": include}, separators=(",", ":")))


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def _render_table(combinations: list[dict[str, str]]) -> str:
    """Return a Markdown view of the matrix, grouped by OS then Python."""
    pretty_os = {"ubuntu-latest": "linux", "macos-latest": "macOS", "macos-26": "macOS"}

    grouped: dict[str, dict[str, list[str]]] = {}
    for r in combinations:
        os_raw = r["os"]
        os_label = pretty_os.get(os_raw, os_raw)
        grouped.setdefault(os_label, {}).setdefault(r["python"], []).append(r["torch"])

    if not grouped:
        return "_No supported combinations yet._\n"

    out: list[str] = []
    for os_label in sorted(grouped):
        out.append(f"### {os_label}")
        out.append("")
        for py in sorted(grouped[os_label], key=_ver_tuple):
            torches = sorted(set(grouped[os_label][py]), key=_ver_tuple)
            out.append(f"- Python {py}: torch {', '.join(torches)}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _splice_in_place(target: Path, table: str) -> str:
    """Replace the block between RENDER_BEGIN/RENDER_END in `target` with `table`.

    Returns the new file contents. Does not write — caller decides.
    """
    text = target.read_text()
    pattern = re.compile(
        re.escape(RENDER_BEGIN) + r".*?" + re.escape(RENDER_END),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        print(
            f"Error: {target} has no {RENDER_BEGIN}/{RENDER_END} block",
            file=sys.stderr,
        )
        sys.exit(1)
    return pattern.sub(RENDER_BEGIN + "\n" + table + RENDER_END, text)


def cmd_render(args) -> None:
    data = load_compat()
    table = _render_table(data["combinations"])

    if args.in_place is None:
        sys.stdout.write(table)
        return

    target = Path(args.in_place)
    new_text = _splice_in_place(target, table)
    if args.check:
        if target.read_text() != new_text:
            print(
                _c(RED, "FAIL") + f": {target} is out of sync; run "
                f"`python scripts/compat_matrix.py render --in-place {target}`",
                file=sys.stderr,
            )
            sys.exit(1)
        print(_c(GREEN, "OK") + f": {target} is in sync")
    else:
        target.write_text(new_text)
        print(_c(GREEN, "✓") + f" rewrote {target}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility matrix manager for neml2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", metavar="command")
    subs.required = True

    subs.add_parser("check", help="validate compatibility.yaml")

    expand_p = subs.add_parser("expand", help="emit JSON matrix for GitHub Actions")
    expand_p.add_argument("--kind", choices=["test", "wheel", "cibuildwheel"], required=True)
    expand_p.add_argument(
        "--mode",
        choices=["full", "pr"],
        default="full",
        help=(
            "full = every combination (push to main / workflow_dispatch); "
            "pr = torch & python extremes per OS (the lean PR subset)."
        ),
    )

    render_p = subs.add_parser("render", help="render the matrix as a Markdown table")
    render_p.add_argument(
        "--in-place",
        metavar="FILE",
        default=None,
        help="splice the table into FILE between BEGIN/END_COMPAT_MATRIX sentinels",
    )
    render_p.add_argument(
        "--check",
        action="store_true",
        help="with --in-place, fail if the file is out of date instead of rewriting",
    )

    args = parser.parse_args()
    if args.command == "check":
        cmd_check(args)
    elif args.command == "expand":
        cmd_expand(args)
    elif args.command == "render":
        cmd_render(args)


if __name__ == "__main__":
    main()
