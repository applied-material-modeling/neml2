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
import shutil


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: update_content.py <content_sources> <dest_dir>")
        sys.exit(1)

    script_name = Path(__file__).name
    src_dir = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])

    # create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    # get current files in the destination directory (remove .in suffix, if any)
    existing_files = set(f.relative_to(dest_dir) for f in dest_dir.rglob("*") if f.is_file())
    existing_files = set(f.with_suffix("") if f.suffix == ".in" else f for f in existing_files)

    # get source files
    source_files = set(f.relative_to(src_dir) for f in src_dir.rglob("*") if f.is_file())

    # remove files that are no longer in the source directory
    obsolete_files = existing_files - source_files
    for f in obsolete_files:
        (dest_dir / f).unlink(missing_ok=True)
        if f.suffix == ".md":
            (dest_dir / f.with_suffix(".md.in")).unlink(missing_ok=True)
        print(f"[Documentation] Removed obsolete source file:", f)
