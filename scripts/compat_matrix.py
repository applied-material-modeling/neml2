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
  compat_matrix.py seed [--max-age-days N]
        regenerate combinations[] in compatibility.yaml from PyPI's torch wheel
        index, filtered to (cp310-cp314, manylinux_x86_64/macosx, last N days)

  compat_matrix.py check
        verify compatibility.yaml internal consistency and sync with
        dependencies.yaml

  compat_matrix.py expand --kind {test|wheel}
        print the matrix as JSON for use in a GitHub Actions strategy.matrix

  compat_matrix.py render [--in-place FILE] [--check]
        render the matrix as a Markdown table; with --in-place, splice it into
        FILE between BEGIN_COMPAT_MATRIX / END_COMPAT_MATRIX sentinels; with
        --check, exit 1 if the in-place rendering would change the file
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install it with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
COMPAT_FILE = REPO_ROOT / "compatibility.yaml"
DEPS_FILE = REPO_ROOT / "dependencies.yaml"

# cibuildwheel build = "cp310-* cp311-* cp312-* cp313-* cp314-*" in pyproject.toml.
# Keep these in sync with [tool.cibuildwheel] build.
SUPPORTED_PYTHONS = {f"cp3{n}": f"3.{n}" for n in range(10, 15)}  # cp310 -> 3.10

# GitHub Actions runner labels paired with the wheel platform tags we care about.
SUPPORTED_OSES = ["ubuntu-latest", "macos-latest"]

# Wheel filename per PEP 427:
#   {dist}-{version}(-{build_tag})?-{python}-{abi}-{platform}.whl
_WHEEL_RE = re.compile(
    r"^(?P<dist>[^-]+)-(?P<version>[^-]+)"
    r"(?:-(?P<build>\d[^-]*))?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^.]+)\.whl$"
)

PYPI_TORCH_SIMPLE = "https://pypi.org/simple/torch/"
PYPI_SIMPLE_ACCEPT = "application/vnd.pypi.simple.v1+json"

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


def _platform_to_os(platform_tag: str) -> str | None:
    """Map a wheel platform tag to a GitHub runner label, or None if unsupported.

    We only care about CPU wheels we can install on the runners we actually have.
    """
    if platform_tag.startswith("manylinux") and platform_tag.endswith("_x86_64"):
        return "ubuntu-latest"
    if platform_tag.startswith("macosx_") and (
        platform_tag.endswith("_arm64") or platform_tag.endswith("_universal2")
    ):
        return "macos-latest"
    return None


def _parse_wheel(filename: str) -> tuple[str, str, str] | None:
    """Return (version, python, os) for a wheel filename we test, else None."""
    m = _WHEEL_RE.match(filename)
    if not m or m.group("dist") != "torch":
        return None
    py_tag = m.group("python")
    if py_tag not in SUPPORTED_PYTHONS:
        return None
    os_label = _platform_to_os(m.group("platform"))
    if os_label is None:
        return None
    return (m.group("version"), SUPPORTED_PYTHONS[py_tag], os_label)


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def _fetch_pypi_torch_files() -> list[dict]:
    """Return the `files` list from PyPI's simple-JSON endpoint for torch."""
    req = urllib.request.Request(PYPI_TORCH_SIMPLE, headers={"Accept": PYPI_SIMPLE_ACCEPT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data.get("files", [])


def _write_compat(build_torch: str, combinations: list[dict]) -> None:
    """Re-emit compatibility.yaml from scratch.

    `seed` is destructive by design: comments other than the fixed header are
    lost. The maintainer expresses prune decisions by removing rows; rationale
    belongs in commit messages.
    """
    lines = [
        "# Compatibility matrix for NEML2 wheels published to PyPI.",
        "#",
        "# Each row in `combinations` asserts: a NEML2 wheel built for the listed",
        "# `python` on `os`, installed alongside `torch == torch`, MUST pass the",
        "# Python test suite (tests).",
        "#",
        "# CI (.github/workflows/compat.yml) enforces this on every PR and push",
        "# to main. Adding a row tightens the supported set; removing one drops",
        "# support.",
        "#",
        "# To regenerate the seeded list from PyPI:",
        "#   python scripts/compat_matrix.py seed",
        "",
        "# The torch version the wheels are compiled against. Mirrors",
        "# torch.version in dependencies.yaml; CI fails if they drift.",
        "# dependencies: torch.version",
        f'build_torch: "{build_torch}"',
        "",
        "combinations:",
    ]
    for row in combinations:
        lines.append(
            f'  - {{ torch: "{row["torch"]}", python: "{row["python"]}", os: {row["os"]} }}'
        )
    COMPAT_FILE.write_text("\n".join(lines) + "\n")


def cmd_seed(args) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.max_age_days)
    print(f"Fetching torch wheel index from PyPI (cutoff {cutoff.date()})...")

    files = _fetch_pypi_torch_files()
    print(f"  fetched {len(files)} entries")

    seen: set[tuple[str, str, str]] = set()
    for f in files:
        if f.get("yanked"):
            continue
        upload_time = f.get("upload-time")
        if not upload_time:
            continue
        # Handle both "...Z" and "...+00:00" forms.
        ts = datetime.fromisoformat(upload_time.replace("Z", "+00:00"))
        if ts < cutoff:
            continue
        parsed = _parse_wheel(f["filename"])
        if parsed is None:
            continue
        seen.add(parsed)

    # Sort by version (PEP 440 ordering via packaging if available; fall back to
    # tuple-of-ints split that's good enough for torch's X.Y.Z scheme).
    def _ver_key(v: str) -> tuple:
        try:
            return tuple(int(p) for p in v.split(".")[:3])
        except ValueError:
            return (0, 0, 0)

    combinations = [
        {"torch": v, "python": p, "os": o}
        for v, p, o in sorted(seen, key=lambda r: (_ver_key(r[0]), r[1], r[2]))
    ]

    existing = load_compat()
    build_torch = existing.get("build_torch") or load_deps()["torch"]["version"]

    _write_compat(build_torch, combinations)

    print(_c(GREEN, "OK") + f": wrote {len(combinations)} rows to {COMPAT_FILE.name}")
    # Quick distribution summary helps the maintainer eyeball what they got.
    by_torch: dict[str, int] = {}
    for row in combinations:
        by_torch[row["torch"]] = by_torch.get(row["torch"], 0) + 1
    for v in sorted(by_torch, key=_ver_key):
        print(f"  torch {v}: {by_torch[v]} rows")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def _ver_tuple(v: str) -> tuple:
    try:
        return tuple(int(p) for p in v.split(".")[:3])
    except ValueError:
        return (0, 0, 0)


def cmd_check(_args) -> None:
    errors: list[str] = []
    data = load_compat()

    deps_torch = load_deps()["torch"]
    build_torch = data.get("build_torch")
    if build_torch != deps_torch["version"]:
        errors.append(
            f"build_torch ({build_torch!r}) != dependencies.yaml torch.version "
            f"({deps_torch['version']!r})"
        )

    v_min = deps_torch.get("version_min")
    v_max = deps_torch.get("version_max")

    valid_pys = set(SUPPORTED_PYTHONS.values())
    for i, row in enumerate(data["combinations"]):
        loc = f"combinations[{i}]"
        for key in ("torch", "python", "os"):
            if key not in row:
                errors.append(f"{loc}: missing '{key}'")
        if row.get("python") not in valid_pys:
            errors.append(f"{loc}: python={row.get('python')!r} not in {sorted(valid_pys)}")
        if row.get("os") not in SUPPORTED_OSES:
            errors.append(f"{loc}: os={row.get('os')!r} not in {SUPPORTED_OSES}")
        # Bounds check: every row must be within the advertised range so the
        # YAML, pyproject.toml constraint, and doc page can't drift.
        t = row.get("torch")
        if t is not None and v_min is not None and _ver_tuple(t) < _ver_tuple(v_min):
            errors.append(f"{loc}: torch={t!r} < version_min={v_min!r}")
        if t is not None and v_max is not None and _ver_tuple(t) > _ver_tuple(v_max):
            errors.append(f"{loc}: torch={t!r} > version_max={v_max!r}")

    # Reject duplicate rows — they'd just cost CI minutes for no signal.
    keys = [(r.get("torch"), r.get("python"), r.get("os")) for r in data["combinations"]]
    seen = set()
    for k in keys:
        if k in seen:
            errors.append(f"duplicate row: {k}")
        seen.add(k)

    if errors:
        print(_c(RED, "FAIL") + f": {len(errors)} issue(s) in {COMPAT_FILE.name}:")
        for e in errors:
            print(f"  {_c(RED, '✗')} {e}")
        sys.exit(1)
    print(
        _c(GREEN, "OK") + f": {len(data['combinations'])} rows, "
        f"build_torch={build_torch}, range=[{v_min}, {v_max}]"
    )


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------


def cmd_expand(args) -> None:
    data = load_compat()
    if args.kind == "test":
        include = [
            {"torch": r["torch"], "python": r["python"], "os": r["os"]}
            for r in data["combinations"]
        ]
    elif args.kind == "wheel":
        seen: set[tuple[str, str]] = set()
        include = []
        for r in data["combinations"]:
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
    pretty_os = {"ubuntu-latest": "linux", "macos-latest": "macOS"}

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

    seed_p = subs.add_parser("seed", help="regenerate combinations from PyPI")
    seed_p.add_argument(
        "--max-age-days",
        type=int,
        default=365,
        help="drop wheels uploaded more than this many days ago (default: 365)",
    )

    subs.add_parser("check", help="validate compatibility.yaml")

    expand_p = subs.add_parser("expand", help="emit JSON matrix for GitHub Actions")
    expand_p.add_argument("--kind", choices=["test", "wheel"], required=True)

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
    if args.command == "seed":
        cmd_seed(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "expand":
        cmd_expand(args)
    elif args.command == "render":
        cmd_render(args)


if __name__ == "__main__":
    main()
