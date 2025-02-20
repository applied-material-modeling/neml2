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

from pathlib import Path
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: extract_tutorial_sources.py tutorials_dir build_dir")
        sys.exit(1)

    tutorials_dir = Path(sys.argv[1])
    if not tutorials_dir.exists():
        print("Tutorial directory not found.")
        sys.exit(1)

    build_dir = Path(sys.argv[2])
    build_dir.mkdir(parents=True, exist_ok=True)

    for md in tutorials_dir.rglob("*.md"):
        current_build_dir = build_dir / md.relative_to(tutorials_dir).parent
        current_build_dir.mkdir(parents=True, exist_ok=True)

        with md.open("r") as f:
            lines = f.readlines()
            srcs = []

            suffix = None
            line_begin = None
            line_end = None
            leading_space = None

            for i, line in enumerate(lines):
                if line_begin:
                    if line.strip().startswith("```"):
                        line_end = i
                        srcs.append((suffix, line_begin, line_end, leading_space))
                        line_begin = None
                        line_end = None
                        suffix = None
                        leading_space = None
                else:
                    if line.strip().startswith("```cpp"):
                        line_begin = i
                        suffix = ".cxx"
                        leading_space = line.find("```cpp")
                    elif line.strip().startswith("```python"):
                        line_begin = i
                        suffix = ".py"
                        leading_space = line.find("```python")

            if suffix or line_begin or line_end or leading_space:
                print("Error: unmatched code block.")
                sys.exit(1)

            count = 0
            for suffix, line_begin, line_end, leading_space in srcs:
                src = current_build_dir / "src_{}{}".format(count, suffix)
                with src.open("w") as f:
                    for line in lines[line_begin + 1 : line_end]:
                        f.write(line[leading_space:])
                count += 1
