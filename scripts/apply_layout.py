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
import xml.etree.ElementTree as ET


def find_child_with_attribute(element, attribute_name, attribute_value):
    """
    Recursively finds a child element with a specific attribute and value.

    Args:
        element: The parent element to search within.
        attribute_name: The name of the attribute to search for.
        attribute_value: The value of the attribute to match.

    Returns:
        The first child element found with the matching attribute and value, or None if not found.
    """
    for child in element:
        if child.get(attribute_name) == attribute_value:
            return child
        # Recursively search in the child's children
        result = find_child_with_attribute(child, attribute_name, attribute_value)
        if result is not None:
            return result
    return None


def insert_title(new_lines, line, navindex):
    ref = line.split(":")[1].strip()
    child = find_child_with_attribute(navindex, "url", "@ref {}".format(ref))
    if child is None:
        print("Error: no navindex entry found for reference {}.".format(ref))
        sys.exit(1)
    new_lines.append("# {} {{#{}}}\n".format(child.get("title"), ref))
    return ref


def get_ref(line):
    return line.split("{#")[1].split("}")[0]


def insert_subsection_list(new_lines, ref, leading_space, navindex):
    child = find_child_with_attribute(navindex, "url", "@ref {}".format(ref))
    if child is None:
        print("Error: no navindex entry found for reference {}.".format(ref))
        sys.exit(1)
    for subchild in child.findall("tab"):
        new_lines.append(" " * leading_space + "- {}\n".format(subchild.get("url")))


def find_adjacent_elements(navindex, element):
    parent = navindex.find(".//{}[@url='{}']...".format(element.tag, element.get("url")))
    if parent is None:
        print("Error: no parent found for element {}.".format(element))
        sys.exit(1)
    children = list(parent)
    index = children.index(element)
    yield children[index - 1] if index > 0 else parent
    yield children[index + 1] if index < len(children) - 1 else parent


def insert_page_navigation(new_lines, ref, navindex):
    nav = """<div class="section_buttons">

| Previous |     Next |
|:---------|---------:|
| {}       |       {} |

</div>
"""
    current = find_child_with_attribute(navindex, "url", "@ref {}".format(ref))
    if current is None:
        print("Error: no navindex entry found for reference {}.".format(ref))
        sys.exit(1)
    prev, next = find_adjacent_elements(navindex, current)
    nav = nav.format(
        prev.get("url") if prev is not None else "",
        next.get("url") if next is not None else "",
    )
    new_lines.append(nav)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: apply_layout.py DoxygenLayout.xml build_dir")
        sys.exit(1)

    doxygen_layout = Path(sys.argv[1])
    if not doxygen_layout.exists():
        print("Doxygen layout file not found.")
        sys.exit(1)
    layout = ET.parse(doxygen_layout)
    root = layout.getroot()
    if not root.tag == "doxygenlayout":
        print("Error: not a Doxygen layout file.")
    if not root[0].tag == "navindex":
        print("Error: no navindex found.")
    navindex = root[0]

    build_dir = Path(sys.argv[2])
    if not build_dir.exists():
        print("Tutorial directory not found.")
        sys.exit(1)

    for tutorial in build_dir.rglob("*.md"):
        new_lines = []
        with open(tutorial, "r") as f:
            lines = f.readlines()
            # Get the page reference
            if lines[0].strip().startswith("@insert-title"):
                ref = insert_title(new_lines, lines[0], navindex)
            else:
                ref = get_ref(lines[0])
                new_lines.append(lines[0])
            # Process directives
            for line in lines[1:]:
                if line.strip() == "@insert-subsection-list":
                    leading_space = len(line) - len(line.lstrip())
                    insert_subsection_list(new_lines, ref, leading_space, navindex)
                elif line.strip() == "@insert-page-navigation":
                    insert_page_navigation(new_lines, ref, navindex)
                else:
                    new_lines.append(line)
        # Write the new file
        with open(tutorial, "w") as f:
            f.writelines(new_lines)
