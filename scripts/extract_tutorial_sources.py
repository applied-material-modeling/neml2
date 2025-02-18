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


def get_src(lines):
    tokens = lines[0].strip().split(":")
    name = tokens[1]
    if not lines[1].strip().startswith("```") or not lines[-1].strip().startswith("```"):
        print("Error: @source and @endsource must enclose a code block.")
        sys.exit(1)
    type = lines[1].strip()[3:]
    if type == "cpp":
        name += ".cxx"
    elif type == "python":
        name += ".py"
    else:
        print("Error: unknown source type {}.".format(type))
        sys.exit(1)
    leading_space = len(lines[2]) - len(lines[2].lstrip())
    new_lines = [line[leading_space:] for line in lines[2:-1]]
    src = "".join(new_lines)
    return name, src


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
            src_blks = []

            src_begin = None

            for i, line in enumerate(lines):
                if src_begin:
                    if line.strip() == "@endsource":
                        src_blks.append((src_begin, i))
                        src_begin = None
                else:
                    if line.strip().startswith("@source:"):
                        src_begin = i

            if src_begin:
                print("Error: unmatched code block.")
                sys.exit(1)

            srcs = {}
            for src_begin, src_end in src_blks:
                name, new_src = get_src(lines[src_begin:src_end])
                src = srcs.setdefault(name, "")
                srcs[name] = src + new_src

            for name, src in srcs.items():
                src_file = current_build_dir / name
                with src_file.open("w") as f:
                    f.write(src)
