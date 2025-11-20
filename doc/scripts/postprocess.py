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

import sys
from pathlib import Path

from utils import *


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: postprocess.py <content_dir>")
        sys.exit(1)

    content_dir = Path(sys.argv[1])
    if not content_dir.exists():
        print("Content directory not found.")
        sys.exit(1)

    for md in content_dir.rglob("*.md"):
        new_lines = []
        with open(md, "r") as f:
            lines = f.readlines()

            # Process directives
            for line in lines:

                # Output
                if line.strip().startswith("@insert-output"):
                    tokens = line.strip().split(":")
                    if len(tokens) < 2:
                        print(f"Invalid @insert-output directive: {line.strip()}")
                        sys.exit(1)
                    output = tokens[1]
                    file = (md.parent / output).with_suffix(".out")
                    if not file.exists():
                        print(f"Output file not found: {file}")
                        sys.exit(1)
                    content = ["```\n"]
                    with open(file, "r") as of:
                        content += of.readlines()
                    content.append("```\n")

                # Regular line
                else:
                    new_lines.append(line)
                    continue

                # Pad leading space and add to new lines
                leading_space = len(line) - len(line.lstrip())
                new_lines.extend(pad_leading_space(content, leading_space))

        # Write the new file
        with open(md, "w") as f:
            f.writelines(new_lines)
