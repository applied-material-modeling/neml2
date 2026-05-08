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
import subprocess
import sys
import argparse


extensions = {".h": "//", ".cxx": "//", ".py": "#", ".sh": "#", ".js": "//"}
additional_files = {}

exclude_dirs = [
    "contrib",
    "cmake",
    "doc/content",
    "doc/tutorials",
    "doc/config",
    "tests/integration",
]
exclude_files = []

rootdir = Path(".")


def should_check(path):
    for exclude_dir in exclude_dirs:
        if Path(rootdir) / Path(exclude_dir) in path.parents:
            return False

    if str(path) in exclude_files:
        return False

    if path.suffix in extensions:
        return True

    if str(path) in additional_files:
        return True

    return False


def generate_copyright_heading(copyright, prefix):
    return [
        (prefix + " " + line.strip() + "\n").replace(prefix + " \n", prefix + "\n")
        for line in copyright.splitlines(True)
    ]


def _is_comment_line(line, prefix):
    return line.startswith(prefix + " ") or line == prefix + "\n"


def find_header_position(lines, prefix):
    """Locate the copyright heading at the top of the file.

    The heading is only ever recognized in this position, in this order:
      [#!shebang]      (optional, any line starting with "#!")
      [blank line]     (optional, only consumed when a shebang is present)
      <comment block>  (the heading itself, or absent)

    This prevents the search from latching onto an unrelated comment block
    further down (e.g. a section header in the middle of the file).

    Returns (insert_at, existing_start, existing_end):
      - insert_at: where a missing heading would be inserted
      - (existing_start, existing_end): half-open range of an existing heading
        block at this position, or (-1, -1) if none is present
    """
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
        if i < len(lines) and lines[i].strip() == "":
            i += 1

    insert_at = i
    if i < len(lines) and _is_comment_line(lines[i], prefix):
        existing_start = i
        while i < len(lines) and _is_comment_line(lines[i], prefix):
            i += 1
        return insert_at, existing_start, i

    return insert_at, -1, -1


def update_heading_ondemand(path, copyright, prefix, modify):
    heading = generate_copyright_heading(copyright, prefix)
    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    insert_at, existing_start, existing_end = find_header_position(lines, prefix)

    if existing_start >= 0 and lines[existing_start:existing_end] == heading:
        return True

    if not modify:
        return False

    if existing_start >= 0:
        new_lines = lines[:existing_start] + heading + lines[existing_end:]
    else:
        prefix_lines = lines[:insert_at]
        suffix_lines = lines[insert_at:]
        block = list(heading)
        if prefix_lines and prefix_lines[-1].strip() != "":
            block.insert(0, "\n")
        if suffix_lines and suffix_lines[0].strip() != "":
            block.append("\n")
        new_lines = prefix_lines + block + suffix_lines

    with path.open("w", encoding="utf-8") as file:
        file.writelines(new_lines)

    print("Corrected copyright heading for " + str(path))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--modify",
        help="Modify the files to have the correct copyright heading",
        action="store_true",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check. If omitted, all git-tracked files are checked.",
    )
    args = parser.parse_args()

    if args.files:
        candidate_files = args.files
    else:
        candidate_files = subprocess.run(
            ["git", "ls-tree", "-r", "HEAD", "--name-only"], capture_output=True, text=True
        ).stdout.splitlines()

    copyright = Path(rootdir / "LICENSE").read_text()
    print("The copyright statement is:\n")
    print(copyright)

    success = True
    for file in candidate_files:
        file_path = Path(file)
        if should_check(file_path):
            if file_path.suffix in extensions:
                prefix = extensions[file_path.suffix]
            elif file_path.name in additional_files:
                prefix = additional_files[file_path.name]
            else:
                sys.exit("Internal error")

            if not update_heading_ondemand(file_path, copyright, prefix, args.modify):
                print(file)
                success = False

    if success:
        print("All files have the correct copyright heading")
    else:
        sys.exit(
            "The above files do NOT contain the correct copyright heading. Use the -m or --modify option to automatically correct them."
        )
