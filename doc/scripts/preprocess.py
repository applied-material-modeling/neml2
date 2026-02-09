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

import layout
import listing

from utils import *


def preprocess(doxygen_layout: Path, mds: list[Path], content_dir: Path):
    """
    Preprocesses markdown files in the content directory using the Doxygen layout.
    """
    navindex = layout.get_navindex(doxygen_layout)

    # sort files so that the behavior is deterministic
    mds.sort()
    for md in mds:
        logger.trace("preprocessing {}", md.relative_to(content_dir))
        new_lines = []
        with open(md, "r") as f:
            lines = f.readlines()

            # Get the page reference
            if lines[0].strip().startswith("@insert-title"):
                new_line, ref = layout.get_title(lines[0], navindex)
                new_lines.append(new_line)
            else:
                ref = layout.get_ref(lines[0])
                new_lines.append(lines[0])

            # Process directives
            for line in lines[1:]:

                # Subsection list
                if line.strip() == "@insert-subsection-list" and ref is not None:
                    content = layout.get_subsection_list(ref, navindex)

                # Page navigation
                elif line.strip() == "@insert-page-navigation" and ref is not None:
                    content = layout.get_page_navigation(ref, navindex)

                # List text
                elif line.strip().startswith("@list:"):
                    tokens = line.strip().split(":")
                    if len(tokens) < 3:
                        logger.error("invalid @list directive: {}", line.strip())
                        sys.exit(1)
                    language = tokens[1]
                    textfile = tokens[2]
                    label = tokens[3] if len(tokens) > 3 else None
                    content = listing.list_text(textfile, language, label)

                # List example output
                elif line.strip().startswith("@list-output:"):
                    tokens = line.strip().split(":")
                    if len(tokens) != 2:
                        logger.error("invalid @list-output directive: {}", line.strip())
                        sys.exit(1)
                    ex = tokens[1]
                    content = listing.list_output(md.parent, ex)

                # List HIT input file
                elif line.strip().startswith("@list-input:"):
                    tokens = line.strip().split(":")
                    ifile = tokens[1]
                    section = tokens[2] if len(tokens) > 2 else None
                    content = listing.list_hit_input(ifile, section)

                # Regular line
                else:
                    new_lines.append(line)
                    continue

                # Pad leading space and add to new lines
                leading_space = len(line) - len(line.lstrip())
                new_lines.extend(pad_leading_space(content, leading_space))

        # Write the new file
        with open(md.with_suffix(""), "w") as f:
            f.writelines(new_lines)
