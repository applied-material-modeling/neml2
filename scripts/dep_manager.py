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

"""One-stop dependency manager for neml2.

Usage:
  dep_manager.py check              verify all files match dependencies.yaml
  dep_manager.py list               list all dependencies and their versions
  dep_manager.py bump DEP.FIELD VALUE  update a dependency version

Annotation convention in source files:
  # dependencies: DEP.FIELD
  <line containing the version value>
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install it with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
DEPS_FILE = REPO_ROOT / "dependencies.yaml"

# Keys in a dep entry that are not version fields
_STRUCTURAL_KEYS = {"files"}

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code, text):
    return f"{code}{text}{RESET}" if _supports_color() else text


def load_deps() -> dict:
    with open(DEPS_FILE) as f:
        return yaml.safe_load(f)


def _value_fields(dep_data: dict) -> dict:
    """Return the scalar fields of a dep entry (version, version_min, etc.)."""
    return {k: v for k, v in dep_data.items() if k not in _STRUCTURAL_KEYS and isinstance(v, str)}


def _find_annotated_lines(filepath: Path, annotation_key: str) -> list[tuple[int, int]]:
    """Return (annotation_idx, value_idx) pairs for each occurrence in the file."""
    lines = filepath.read_text().splitlines()
    markers = {
        f"# dependencies: {annotation_key}",
        f"<!-- dependencies: {annotation_key} -->",
    }
    results = []
    for i, line in enumerate(lines):
        if line.strip() in markers:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                results.append((i, j))
    return results


def _update_yaml_field(dep_name: str, field: str, old_val: str, new_val: str) -> None:
    """Replace old_val with new_val for field under dep_name in dependencies.yaml."""
    lines = DEPS_FILE.read_text().splitlines(keepends=True)
    in_dep = False
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped == f"{dep_name}:":
            in_dep = True
        elif in_dep and stripped and not stripped[0].isspace() and not stripped.startswith("#"):
            in_dep = False
        elif in_dep and re.match(rf"^\s+{re.escape(field)}:", line):
            if old_val in line:
                lines[i] = line.replace(old_val, new_val, 1)
            break
    DEPS_FILE.write_text("".join(lines))


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def cmd_check(deps: dict, _args) -> None:
    """Scan all listed files for # dependencies: DEP.FIELD annotations and validate them.

    A file is allowed to have annotations for only a subset of a dep's fields — the
    check validates every annotation that IS present, but does not require all fields
    to appear in every listed file.
    """
    errors = []
    ok_count = 0

    for dep_name, dep_data in deps.items():
        fields = _value_fields(dep_data)
        files = dep_data.get("files", [])

        for filepath in files:
            full_path = REPO_ROOT / filepath
            if not full_path.exists():
                errors.append(f"{dep_name}: file not found: {filepath}")
                continue

            file_lines = full_path.read_text().splitlines()

            for field, value in fields.items():
                annotation_key = f"{dep_name}.{field}"
                occurrences = _find_annotated_lines(full_path, annotation_key)
                for _, j in occurrences:
                    if value not in file_lines[j]:
                        errors.append(
                            f"{annotation_key}: {filepath}:{j + 1}: "
                            f"expected '{value}', got: '{file_lines[j].strip()}'"
                        )
                    else:
                        ok_count += 1

    if errors:
        count = len(errors)
        label = "inconsistency" if count == 1 else "inconsistencies"
        print(_c(RED, "FAIL") + f": {count} {label} found:")
        for e in errors:
            print(f"  {_c(RED, '✗')} {e}")
        sys.exit(1)
    else:
        print(_c(GREEN, "OK") + f": all {ok_count} annotations are consistent")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def cmd_list(deps: dict, _args) -> None:
    col_dep = 16
    col_field = 13

    rows = []
    for dep_name, dep_data in deps.items():
        for field, value in _value_fields(dep_data).items():
            rows.append((dep_name, field, value))

    col_val = max((len(r[2]) for r in rows), default=0)
    header = f"{'dep':<{col_dep}}  {'field':<{col_field}}  value"
    print(_c(BOLD, header))
    print("─" * (col_dep + col_field + col_val + 6))
    for dep_name, field, value in rows:
        print(f"{dep_name:<{col_dep}}  {field:<{col_field}}  {value}")


# ---------------------------------------------------------------------------
# bump
# ---------------------------------------------------------------------------


def cmd_bump(deps: dict, args) -> None:
    spec: str = args.spec
    new_val: str = args.value
    dry_run: bool = args.dry_run

    if "." not in spec:
        print("Error: spec must be DEP.FIELD (e.g. catch2.version)", file=sys.stderr)
        sys.exit(1)

    dep_name, field = spec.split(".", 1)

    if dep_name not in deps:
        print(f"Error: unknown dependency '{dep_name}'", file=sys.stderr)
        sys.exit(1)

    dep_data = deps[dep_name]

    # --- regular field ---
    value_fields = _value_fields(dep_data)
    if field not in value_fields:
        print(
            f"Error: '{dep_name}' has no field '{field}'. Available: {list(value_fields)}",
            file=sys.stderr,
        )
        sys.exit(1)

    old_val = dep_data[field]
    if old_val == new_val:
        print(f"{dep_name}.{field} is already at {old_val}")
        return

    annotation_key = f"{dep_name}.{field}"
    files = dep_data.get("files", [])

    updated_files = []
    skipped = []
    for filepath in files:
        full_path = REPO_ROOT / filepath
        if not full_path.exists():
            skipped.append(filepath)
            continue

        file_lines = full_path.read_text().splitlines(keepends=True)
        occurrences = _find_annotated_lines(full_path, annotation_key)
        changed = False
        for _, j in occurrences:
            if old_val in file_lines[j]:
                changed = True
                if not dry_run:
                    file_lines[j] = file_lines[j].replace(old_val, new_val, 1)
        if changed:
            if not dry_run:
                full_path.write_text("".join(file_lines))
            updated_files.append(filepath)

    if not dry_run:
        _update_yaml_field(dep_name, field, old_val, new_val)

    label = _c(YELLOW, "dry-run") if dry_run else _c(GREEN, "✓")
    print(f"{label} {annotation_key}: {old_val} → {new_val}")
    verb = "would update" if dry_run else "updated"
    for f in updated_files:
        print(f"  {_c(GREEN, verb)} {f}")
    for f in skipped:
        print(f"  {_c(YELLOW, 'skipped')} {f} (not found)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dependency manager for neml2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", metavar="command")
    subs.required = True

    subs.add_parser("check", help="verify all files match dependencies.yaml")
    subs.add_parser("list", help="list all dependencies and their versions")

    bump_p = subs.add_parser("bump", help="update a dependency version")
    bump_p.add_argument("spec", help="DEP.FIELD (e.g. catch2.version)")
    bump_p.add_argument("value", help="new version value")
    bump_p.add_argument(
        "--dry-run", action="store_true", help="show what would change without writing"
    )

    args = parser.parse_args()
    deps = load_deps()

    if args.command == "check":
        cmd_check(deps, args)
    elif args.command == "list":
        cmd_list(deps, args)
    elif args.command == "bump":
        cmd_bump(deps, args)


if __name__ == "__main__":
    main()
