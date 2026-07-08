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
  dep_manager.py get DEP.FIELD      print the bare value of one DEP.FIELD
  dep_manager.py bump DEP.FIELD VALUE  update a dependency version
  dep_manager.py sync [--source F]  adopt pins edited in F (default pyproject.toml)

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
DEPS_FILE = Path(__file__).parent / "dependencies.yaml"

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


def _is_annotation_line(line: str) -> bool:
    """True if the line is any `# dependencies: ...` / `// dependencies: ...` /
    HTML-comment annotation."""
    stripped = line.strip()
    return (
        stripped.startswith("# dependencies:")
        or stripped.startswith("// dependencies:")
        or (stripped.startswith("<!-- dependencies:") and stripped.endswith("-->"))
    )


def _find_annotated_lines(filepath: Path, annotation_key: str) -> list[tuple[int, int]]:
    """Return (annotation_idx, value_idx) pairs for each occurrence in the file.

    Empty lines and other dependency-annotation lines between the marker and
    the value are skipped, so multiple annotations can stack on the same value
    line — e.g. `# dependencies: torch.version_min` and
    `# dependencies: torch.version_max` both pointing at
    `"torch>=2.10.0,<=2.12.0"`.
    """
    lines = filepath.read_text().splitlines()
    markers = {
        f"# dependencies: {annotation_key}",
        f"// dependencies: {annotation_key}",
        f"<!-- dependencies: {annotation_key} -->",
    }
    results = []
    for i, line in enumerate(lines):
        if line.strip() in markers:
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or _is_annotation_line(lines[j])):
                j += 1
            if j < len(lines):
                results.append((i, j))
    return results


def _propagate(
    dep_name: str,
    field: str,
    old_val: str,
    new_val: str,
    files: list[str],
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    """Replace old_val with new_val on every annotated line across *files* + the YAML.

    Shared by ``bump`` and ``sync``: both ultimately mean "this dep.field moved
    from old_val to new_val everywhere it is mirrored." Returns
    ``(updated_files, skipped_files)``.
    """
    annotation_key = f"{dep_name}.{field}"
    updated_files: list[str] = []
    skipped: list[str] = []
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

    return updated_files, skipped


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
# get
# ---------------------------------------------------------------------------


def cmd_get(deps: dict, args) -> None:
    """Print the bare value of a single DEP.FIELD, for scripts/CI.

    ``list`` prints a formatted table; this prints just the value (no color, no
    padding) so shell can capture it, e.g.
    ``v$(dep_manager.py get neml2.version | cut -d. -f1,2)``. Errors go to stderr
    with a non-zero exit so a bad key fails the caller loudly.
    """
    spec: str = args.spec

    if "." not in spec:
        print("Error: spec must be DEP.FIELD (e.g. neml2.version)", file=sys.stderr)
        sys.exit(1)

    dep_name, field = spec.split(".", 1)

    if dep_name not in deps:
        print(f"Error: unknown dependency '{dep_name}'", file=sys.stderr)
        sys.exit(1)

    value_fields = _value_fields(deps[dep_name])
    if field not in value_fields:
        print(
            f"Error: '{dep_name}' has no field '{field}'. Available: {list(value_fields)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(value_fields[field])


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

    files = dep_data.get("files", [])
    updated_files, skipped = _propagate(dep_name, field, old_val, new_val, files, dry_run)

    label = _c(YELLOW, "dry-run") if dry_run else _c(GREEN, "✓")
    print(f"{label} {dep_name}.{field}: {old_val} → {new_val}")
    verb = "would update" if dry_run else "updated"
    for f in updated_files:
        print(f"  {_c(GREEN, verb)} {f}")
    for f in skipped:
        print(f"  {_c(YELLOW, 'skipped')} {f} (not found)")


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------

# A PEP 440 / GitHub-tag style version token: one or more dot-separated number
# groups, optional leading ``v``, optional trailing pre/post-release suffix.
# Requires at least one dot so bare integers (e.g. an unrelated ``3``) don't
# masquerade as versions.
_VERSION_RE = re.compile(r"v?\d+(?:\.\d+)+(?:[A-Za-z0-9.\-]*)?")


def _extract_version(line: str, old_val: str) -> str | None:
    """Pull the version literal out of an annotated *line*.

    ``old_val`` is the value currently recorded in dependencies.yaml. If it is
    still present the line is unchanged and we return it verbatim. Otherwise we
    look for exactly one version-shaped token (the one an external tool such as
    Dependabot just rewrote). Returns ``None`` when the line is ambiguous
    (zero or multiple candidate tokens) so the caller can warn rather than guess.
    """
    if old_val in line:
        return old_val
    matches = _VERSION_RE.findall(line)
    return matches[0] if len(matches) == 1 else None


def cmd_sync(deps: dict, args) -> None:
    """Make dependencies.yaml (and all mirrors) agree with a *source* file.

    External tooling -- chiefly Dependabot -- edits a single pinned version in
    one file (``pyproject.toml`` by default) without knowing dependencies.yaml
    is the source of truth. ``sync`` reads the version each tracked dep now has
    in that file and, where it differs from dependencies.yaml, propagates the
    new value to the YAML and every other mirror site (so e.g. a Dependabot
    ``ruff`` bump also lands in ``.pre-commit-config.yaml``). The result is what
    ``dep_manager.py bump`` would have produced, so ``check`` passes afterward.
    """
    source: str = args.source
    dry_run: bool = args.dry_run

    full_source = REPO_ROOT / source
    if not full_source.exists():
        print(f"Error: source file not found: {source}", file=sys.stderr)
        sys.exit(1)

    src_lines = full_source.read_text().splitlines()

    # Detect every intended change up front (before touching any file) so that
    # propagating one dep can't perturb detection of another.
    pending: dict[tuple[str, str], tuple[str, str]] = {}
    warnings: list[str] = []
    for dep_name, dep_data in deps.items():
        if source not in dep_data.get("files", []):
            continue
        for field, yaml_val in _value_fields(dep_data).items():
            for _, j in _find_annotated_lines(full_source, f"{dep_name}.{field}"):
                found = _extract_version(src_lines[j], yaml_val)
                if found is None:
                    warnings.append(
                        f"{dep_name}.{field}: {source}:{j + 1}: "
                        f"could not parse a version from: {src_lines[j].strip()!r}"
                    )
                    continue
                if found != yaml_val:
                    key = (dep_name, field)
                    if key in pending and pending[key][1] != found:
                        warnings.append(
                            f"{dep_name}.{field}: conflicting values in {source} "
                            f"({pending[key][1]} vs {found}); skipping"
                        )
                        pending[key] = (yaml_val, yaml_val)  # neutralize
                    else:
                        pending[key] = (yaml_val, found)

    pending = {k: v for k, v in pending.items() if v[0] != v[1]}

    for w in warnings:
        print(_c(YELLOW, "warning") + f": {w}", file=sys.stderr)

    if not pending:
        print(_c(GREEN, "OK") + f": dependencies.yaml already in sync with {source}")
        return

    for (dep_name, field), (old_val, new_val) in pending.items():
        files = deps[dep_name].get("files", [])
        updated_files, skipped = _propagate(dep_name, field, old_val, new_val, files, dry_run)
        label = _c(YELLOW, "dry-run") if dry_run else _c(GREEN, "✓")
        print(f"{label} {dep_name}.{field}: {old_val} → {new_val}")
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

    get_p = subs.add_parser("get", help="print the bare value of a single DEP.FIELD")
    get_p.add_argument("spec", help="DEP.FIELD (e.g. neml2.version)")

    bump_p = subs.add_parser("bump", help="update a dependency version")
    bump_p.add_argument("spec", help="DEP.FIELD (e.g. catch2.version)")
    bump_p.add_argument("value", help="new version value")
    bump_p.add_argument(
        "--dry-run", action="store_true", help="show what would change without writing"
    )

    sync_p = subs.add_parser(
        "sync", help="adopt version pins edited in a source file (e.g. by Dependabot)"
    )
    sync_p.add_argument(
        "--source",
        default="pyproject.toml",
        help="file whose pinned versions to adopt (default: pyproject.toml)",
    )
    sync_p.add_argument(
        "--dry-run", action="store_true", help="show what would change without writing"
    )

    args = parser.parse_args()
    deps = load_deps()

    if args.command == "check":
        cmd_check(deps, args)
    elif args.command == "list":
        cmd_list(deps, args)
    elif args.command == "get":
        cmd_get(deps, args)
    elif args.command == "bump":
        cmd_bump(deps, args)
    elif args.command == "sync":
        cmd_sync(deps, args)


if __name__ == "__main__":
    main()
