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

"""Apply lcov exclusion markers to an llvm-cov-generated lcov file, in place.

``llvm-cov`` (clang source-based coverage) does not honor the in-source lcov
exclusion comments the way ``geninfo`` does, so this post-processor implements
them for the ``cpp_coverage.sh`` pipeline. Recognized markers (matched anywhere
on a source line, per the lcov convention):

- ``LCOV_EXCL_LINE``  -- exclude that single line
- ``LCOV_EXCL_START`` -- begin an excluded block (the marker line included)
- ``LCOV_EXCL_STOP``  -- end an excluded block (the marker line included)

For every excluded source line we drop its per-line (``DA:``) and per-branch
(``BRDA:``) records so the line disappears from coverage entirely (neither hit
nor miss nor partial). Function records and the summary counters are left as-is;
Codecov recomputes totals from the remaining ``DA``/``BRDA`` records.

Usage: ``lcov_apply_exclusions.py <src_root> <coverage.lcov>``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_DA = re.compile(r"^(?:DA|BRDA):(\d+),")


def excluded_lines(source: Path) -> set[int]:
    """1-based line numbers excluded by lcov markers in `source` (empty if absent)."""
    excl: set[int] = set()
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return excl
    in_block = False
    for i, line in enumerate(text.splitlines(), start=1):
        start = "LCOV_EXCL_START" in line
        stop = "LCOV_EXCL_STOP" in line
        if start:
            in_block = True
        if in_block or start or stop or "LCOV_EXCL_LINE" in line:
            excl.add(i)
        if stop:
            in_block = False
    return excl


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: lcov_apply_exclusions.py <src_root> <coverage.lcov>", file=sys.stderr)
        return 2
    src_root = Path(argv[1])
    lcov = Path(argv[2])

    out: list[str] = []
    current: set[int] = set()
    cache: dict[str, set[int]] = {}
    dropped = 0
    for line in lcov.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("SF:"):
            sf = line[3:].strip()
            path = Path(sf) if Path(sf).is_absolute() else src_root / sf
            key = str(path)
            if key not in cache:
                cache[key] = excluded_lines(path)
            current = cache[key]
            out.append(line)
            continue
        m = _DA.match(line)
        if m and int(m.group(1)) in current:
            dropped += 1
            continue  # drop the excluded line's coverage record
        out.append(line)

    lcov.write_text("".join(out), encoding="utf-8")
    print(f"lcov_apply_exclusions: dropped {dropped} record(s) for LCOV_EXCL-marked lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
