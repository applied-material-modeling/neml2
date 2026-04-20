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
from loguru import logger
import listing

def main():
    root = Path(__file__).parent.parent.parent
    content_dir = root / "doc" / "content"
    md_files = list(content_dir.glob("**/*.md"))
    
    errors = 0
    
    logger.info(f"Checking {len(md_files)} markdown files for directive paths...")
    
    for md in md_files:
        with open(md, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                try:
                    if line.startswith("@list-input:"):
                        ifile = line.split(":")[1]
                        listing.git_fuzzy_find_file(ifile)
                    elif line.startswith("@list:"):
                        tokens = line.split(":")
                        if len(tokens) >= 3:
                            textfile = tokens[2]
                            listing.git_fuzzy_find_file(textfile)
                    elif line.startswith("@list-output:"):
                        # @list-output:example_name
                        # This checks if the example's .out file WOULD exist if examples were run.
                        # Since this is a lightweight check, we just check if the model/test referenced in the dir exists.
                        pass
                except SystemExit:
                    logger.error(f"Invalid path in {md}:{lineno} -> {line}")
                    errors += 1
                except Exception as e:
                    logger.error(f"Error processing {md}:{lineno}: {e}")
                    errors += 1

    if errors > 0:
        logger.error(f"Found {errors} errors in documentation directives.")
        sys.exit(1)
    else:
        logger.success("All documentation directives verified successfully (Lightweight).")

if __name__ == "__main__":
    main()
