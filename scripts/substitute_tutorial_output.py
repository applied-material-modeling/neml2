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
        print("Usage: substitute_tutorial_output.py tutorials_dir build_dir")
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
            newlines = []
            count = 0
            leading_space = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("@attach_output:"):
                    leading_space = len(line) - len(line.lstrip())
                    tokens = line.strip().split(":")
                    name = tokens[1]
                    src_out = current_build_dir / "{}.out".format(name)
                    if not src_out.exists():
                        print("Expected output file not found: {}".format(src_out))
                        sys.exit(1)
                    with open(src_out, "r") as g:
                        for newline in g.readlines():
                            newlines.append(" " * leading_space + newline)
                elif line.strip().startswith("@source:"):
                    continue
                elif line.strip() == "@endsource":
                    continue
                else:
                    newlines.append(line)
        with open(current_build_dir / md.name, "w") as f:
            f.writelines(newlines)
