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


def get_first_comment_block(path, prefix):
    # Find the starting and ending line numbers of the first comment block
    # (It's okay if the license header doesn't start from the first line.
    #  This relaxation is useful for things like shebang which have to be on the first line.)
    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()
        state = 0
        lineno_start = -1
        lineno_end = -1
        for i, line in enumerate(lines):
            if state == 0 and (line.startswith(prefix + " ") or line == prefix + "\n"):
                lineno_start = i
                state = 1
            elif state == 1 and not (line.startswith(prefix + " ") or line == prefix + "\n"):
                lineno_end = i
                break

    return lines, lineno_start, lineno_end


def has_correct_heading(lines, start, end, correct_heading):
    if start < 0:
        return False
    if end <= start:
        return lines[start:] == correct_heading
    if end > start:
        return lines[start:end] == correct_heading
    return False


def update_heading(path, lines, start, end, correct_heading):
    if start >= 0:
        if end > start:
            newlines = lines[:start] + correct_heading + lines[end:]
        else:
            newlines = lines[:start] + correct_heading
    else:
        newlines = correct_heading

    with path.open("w", encoding="utf-8") as file:
        for line in newlines:
            file.write(line)

    print("Corrected copyright heading for " + str(path))


def update_heading_ondemand(path, copyright, prefix, modify):
    heading = generate_copyright_heading(copyright, prefix)
    lines, lineno_start, lineno_end = get_first_comment_block(path, prefix)
    correct = has_correct_heading(lines, lineno_start, lineno_end, heading)

    if not modify:
        return correct

    if correct:
        return True

    update_heading(path, lines, lineno_start, lineno_end, heading)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--modify",
        help="Modify the files to have the correct copyright heading",
        action="store_true",
    )
    args = parser.parse_args()

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

    files = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"], capture_output=True, text=True
    ).stdout

    copyright = Path(rootdir / "LICENSE").read_text()
    print("The copyright statement is:\n")
    print(copyright)

    success = True
    for file in files.splitlines():
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
