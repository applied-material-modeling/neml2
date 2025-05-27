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
import argparse
import yaml

comment_prefix = {
    "toml": "#",
    "yaml": "#",
    "txt": "#",
    "cmake.in": "#",
    "md": "<!--",
}

if __name__ == "__main__":
    # cliargs
    parser = argparse.ArgumentParser(description="Utilities for managing dependencies.")
    parser.add_argument(
        "-d",
        "--dependencies",
        type=str,
        default="dependencies.yaml",
        help="The file containing the dependencies in YAML format.",
    )

    # parse cliargs
    args = parser.parse_args()
    root = Path(__file__).parent.parent.resolve()
    dep_file = root / args.dependencies
    deps = yaml.safe_load(dep_file.read_text())

    # check dependencies
    ret = 0
    for dep, specs in deps.items():
        print(dep)
        for file in specs.get("files", []):
            file_path = root / file

            print("  file:", file_path.relative_to(root))

            if not file_path.exists():
                print("  \033[31merror: File does not exist.\033[0m")
                ret = 1
                continue

            _, _, suffix = file_path.name.partition(".")
            if suffix not in comment_prefix:
                print(
                    f"  \033[31merror: File {file_path} has an unsupported extension {suffix}.\033[0m"
                )
                ret = 1
                continue

            prefix = comment_prefix[suffix]
            with file_path.open("r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith(prefix):
                        line = line[len(prefix) :].strip()
                        tokens = line.split()
                        for token in tokens:
                            if token.split(".")[0] == dep:
                                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                                for key in token.split(".")[1:]:
                                    expected = str(specs.get(key, None))
                                    if not expected:
                                        break
                                if not expected:
                                    print(
                                        f"    \033[31merror: Unable to parse specification '{token}'.\033[0m"
                                    )
                                    ret = 1
                                    continue
                                if not expected in next_line:
                                    print(
                                        f"    \033[31merror: Expected version specification not found.\033[0m"
                                    )
                                    ret = 1
                                next_line_highlighted = next_line.replace(
                                    expected, f"\033[92m{expected}\033[0m"
                                )
                                print(f"    line {i+1}: {next_line_highlighted}")

        print()
    exit(ret)
