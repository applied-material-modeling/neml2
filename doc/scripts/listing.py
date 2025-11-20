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
import re


def git_fuzzy_find_file(filename: str) -> Path:
    """
    Uses git to fuzzy find a file in the repository.

    Args:
        filename: The name of the file to find.

    Returns:
        The Path to the file if found, None otherwise.
    """
    import subprocess

    try:
        root = Path(__file__).parent.parent
        result = subprocess.run(
            ["git", "ls-files", "*{}".format(filename)],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        )
        files = result.stdout.strip().split("\n")
        if len(files) > 1:
            print(f"Multiple files found for '{filename}': {files}")
            sys.exit(1)
        if not files or not files[0]:
            print(f"No files found for '{filename}'.")
            sys.exit(1)
        return root / files[0]
    except subprocess.CalledProcessError:
        print(f"Error running command: git ls-files {filename}")
        sys.exit(1)


SECTION_OPEN_RE = re.compile(r"\[(.+?)\]")  # matches [Name], captures Name


def parse_input(input: Path) -> dict[str, tuple[int, int | None, dict]]:
    """
    Parses a HIT input file to extract its section structure.
    Args:
        input: Path to the HIT input file.
    Returns:
        A nested dictionary representing the section structure. Dictionary keys are section names,
        and values are tuples of (start_line, end_line, children_dict).
    """

    with open(input, "r") as f:
        lines = f.readlines()

    # Internal representation for each node:
    # { "start": int, "end": int | None, "children": {name: node, ...} }
    root_children = {}
    # Stack of (name, node_dict). Root has a dummy name and holds root_children.
    stack = [("ROOT", {"start": None, "end": None, "children": root_children})]

    for lineno, raw_line in enumerate(lines, start=1):  # 1-based line numbers
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Closing a section: line is exactly []
        if line == "[]":
            if len(stack) == 1:
                raise ValueError(f"Unmatched [] at line {lineno}")
            name, node = stack.pop()
            node["end"] = lineno
            continue

        # Opening a section: [Name]
        m = SECTION_OPEN_RE.fullmatch(line)
        if m:
            sec_name = m.group(1).strip()
            parent_node = stack[-1][1]  # current top of stack
            children = parent_node["children"]  # its children dict
            if sec_name in children:
                print(f"Duplicate section name '{sec_name}' under the same parent at line {lineno}")
                sys.exit(1)
            new_node = {"start": lineno, "end": None, "children": {}}
            children[sec_name] = new_node
            stack.append((sec_name, new_node))
            continue

        # any other line is just content, ignore for structure

    if len(stack) != 1:
        # some sections not closed
        open_sections = [name for name, _ in stack[1:]]
        raise ValueError(f"Unclosed sections at EOF: {open_sections}")

    # Convert internal node representation to the requested tuple format
    def to_tuple_dict(children_dict):
        out = {}
        for name, node in children_dict.items():
            out[name] = (
                node["start"],
                node["end"],
                to_tuple_dict(node["children"]),
            )
        return out

    return to_tuple_dict(root_children)


def list_hit_section(
    structure: dict, lines: list[str], section_map: dict[str, dict], top_level: bool = True
) -> str:
    content = ""
    for section, subsections in section_map.items():
        if section not in structure:
            print(f"Section '{section}' not found.")
            sys.exit(1)
        start, end, substructure = structure[section]
        if end is None:
            print(f"Section '{section}' is not closed.")
            sys.exit(1)

        if not subsections:
            return "".join(lines[start - 1 : end])

        content += lines[start - 1]  # include section header line
        content += list_hit_section(substructure, lines, subsections, top_level=False)
        content += lines[end - 1]  # include section closing line
        content += "\n" if top_level else ""
    return content


def build_nested_dict(paths, sep="/"):
    root = {}

    for path in paths:
        parts = path.split(sep)
        d = root
        for p in parts:
            d = d.setdefault(p, {})
    return root


def list_hit_input(ifile: str, spec: str | None = None) -> list[str]:
    """
    Lists the contents of a HIT input file or a specific section.

    Args:
        ifile: The HIT input file name.
        spec: A comma-delimited list of section names to list, or None for the whole file.
    Returns:
        A string representing the markdown code block of the HIT input file or section.
    """
    input_path = git_fuzzy_find_file(ifile)
    structure = parse_input(input_path)
    with open(input_path, "r") as f:
        lines = f.readlines()

    if spec is None:
        # Return the whole file
        content = "".join(lines)
    else:
        # Create the section map to list
        sections = spec.split(",")
        section_map = build_nested_dict(sections)
        content = list_hit_section(structure, lines, section_map)

    # Return as a markdown code block
    content = f"```\n{content}```\n"
    return content.splitlines(keepends=True)


def list_text(file: str, language: str, start: int | None, end: int | None) -> list[str]:
    """
    Lists the contents of a text file or a specific line range.

    Args:
        file: The text file name.
        start: The starting line number (1-based), or None for the beginning.
        end: The ending line number (1-based, inclusive), or None for the end.
    Returns:
        A string representing the contents of the specified line range.
    """
    path = git_fuzzy_find_file(file)
    with open(path, "r") as f:
        lines = f.readlines()

    start = start or 1
    end = end or len(lines)
    content = "".join(lines[start - 1 : end])
    content = f"```{language}\n{content}```\n"
    return content.splitlines(keepends=True)
