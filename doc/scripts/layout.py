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
from typing import Union


def get_navindex(doxygen_layout: Path) -> ET.Element:
    """
    Parses the Doxygen layout XML file and returns the root navindex element.
    """
    layout = ET.parse(doxygen_layout)
    root = layout.getroot()
    if not root.tag == "doxygenlayout":
        print("Error: not a Doxygen layout file.")
    if not root[0].tag == "navindex":
        print("Error: no navindex found.")
    return root[0]


def find_child_with_attribute(
    element: ET.Element, attribute_name: str, attribute_value: str
) -> Union[ET.Element, None]:
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


def get_title(line: str, navindex: ET.Element) -> tuple[str, str]:
    """
    Get the title for the page based on the reference in the line.

    Args:
        line: The line containing the @insert-title directive. Syntax: @insert-title:<page_ref>
        navindex: The root navindex element from the Doxygen layout XML.

    Returns:
        A tuple containing the page reference and the new line with the title.
    """
    ref = line.split(":")[1].strip()
    child = find_child_with_attribute(navindex, "url", "@ref {}".format(ref))
    if child is None:
        print("Error: no navindex entry found for reference {}.".format(ref))
        sys.exit(1)
    new_line = "# {} {{#{}}}\n".format(child.get("title"), ref)
    return new_line, ref


def get_ref(line: str) -> str:
    """
    Extracts the page reference from a line containing a markdown header with an anchor.

    Args:
        line: The line containing the markdown header with an anchor. Syntax: # Title {#page_ref}

    Returns:
        The extracted page reference.
    """
    return line.split("{#")[1].split("}")[0]


def get_subsection_list(ref: str, navindex: ET.Element) -> list[str]:
    """
    Get a subsection list for the given page reference.

    Args:
        ref: The page reference to find subsections for.
        leading_space: The number of leading spaces to indent the list items.
        navindex: The root navindex element from the Doxygen layout XML.

    Returns:
        A list of strings representing the new lines for the subsection list.
    """
    child = find_child_with_attribute(navindex, "url", "@ref {}".format(ref))
    if child is None:
        print("Error: no navindex entry found for reference {}.".format(ref))
        sys.exit(1)
    new_lines = []
    for subchild in child.findall("tab"):
        new_lines.append("- {}\n".format(subchild.get("url")))
    return new_lines


def find_adjacent_elements(navindex: ET.Element, element: ET.Element):
    parent = navindex.find(".//{}[@url='{}']...".format(element.tag, element.get("url")))
    if parent is None:
        print("Error: no parent found for element {}.".format(element))
        sys.exit(1)
    children = list(parent)
    index = children.index(element)
    yield children[index - 1] if index > 0 else parent
    yield children[index + 1] if index < len(children) - 1 else parent


def get_page_navigation(ref: str, navindex: ET.Element) -> list[str]:
    """
    Get page navigation links for the given page reference.

    Args:
        ref: The page reference to find navigation links for.
        navindex: The root navindex element from the Doxygen layout XML.

    Returns:
        A string representing the new lines for the page navigation.
    """

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
    return nav.splitlines(keepends=True)
