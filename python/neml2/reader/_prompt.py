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

from typing import Optional

from ._parser import ModelEntry
from ._syntax import SyntaxDB

_SYSTEM_PROMPT = """\
You are an expert in the NEML2 computational materials science library. \
NEML2 is used to define constitutive material models which mathematically \
describe how materials respond to deformation, temperature, and other physical conditions. \
You will be given a structured description of a NEML2 input file. \
Your task is to explain, in clear and accessible natural language, what material model \
the input file defines, what physics it captures, and how the different components \
work together. Assume the reader is a materials engineer or scientist who understands \
continuum mechanics but may not be familiar with NEML2's input syntax.\
The response should contain:\
1. A high-level summary of the material model and its intended use case. \
2. An explanation of the key components (e.g. hardening rules, flow rules) and how they interact. \
3. A description of any parameters set in the input and their physical significance. \
4. Any assumptions, limitations, or special features of the model that are evident from the input.\
The explanation should be concise but informative, aiming for clarity and insight rather than an \
exhaustive line-by-line commentary. Use analogies or examples where helpful to illustrate complex concepts.\
"""


def _format_entry(entry: ModelEntry, db: SyntaxDB, include_params: bool, indent: str = "") -> str:
    lines = []
    lines.append(f'{indent}Model "{entry.name}" (type: {entry.type})')

    info = db.lookup(entry.type)

    if info is not None:
        if info.doc:
            lines.append(f"{indent}  Description: {info.doc}")
        else:
            lines.append(f"{indent}  Description: (no description available)")
    elif db.available:
        lines.append(f"{indent}  Description: (type not found in syntax database)")

    if include_params:
        user_params = {k: v for k, v in entry.params.items() if k != "type"}
        if user_params:
            lines.append(f"{indent}  Parameters set in input:")
            for k, v in user_params.items():
                lines.append(f"{indent}    {k} = {v}")

    if include_params and info is not None and info.params:
        lines.append(f"{indent}  Parameter reference:")
        param_by_name = {p.name: p for p in info.params}
        for k in entry.params:
            if k == "type":
                continue
            p = param_by_name.get(k)
            if p and p.doc:
                lines.append(f"{indent}    {k} ({p.ftype}): {p.doc}")

    if entry.children:
        for child_sec, child_entries in entry.children.items():
            lines.append(f"{indent}  Subsection: {child_sec}")
            for child in child_entries:
                lines.append(_format_entry(child, db, include_params, indent + "    "))

    return "\n".join(lines)


def build_prompt(
    parsed: dict,
    db: SyntaxDB,
    sections: Optional[list] = None,
    include_params: bool = True,
    system_context: Optional[str] = None,
) -> tuple:
    """
    Build a ``(system_prompt, user_prompt)`` pair from a parsed input file.

    Args:
        parsed: Output of :func:`~neml2.reader.parse_input`.
        db: A :class:`~neml2.reader.SyntaxDB` instance.
        sections: If given, only include these top-level section names
            (e.g. ``["Models"]``). Defaults to all sections.
        include_params: Whether to include per-parameter details.
        system_context: Optional extra text appended to the system prompt.

    Returns:
        A ``(system_prompt, user_prompt)`` tuple of strings.
    """
    system = _SYSTEM_PROMPT
    if system_context:
        system = system + "\n\n" + system_context

    parts = ["NEML2 Input File Analysis", "=" * 25, ""]

    if not db.available:
        parts.append("(Note: Syntax database not available; type descriptions are omitted.)\n")

    target_sections = sections if sections is not None else list(parsed.keys())

    for sec_name in target_sections:
        entries = parsed.get(sec_name)
        if not entries:
            continue

        parts.append(f"Section: {sec_name}")
        parts.append("-" * (len("Section: ") + len(sec_name)))

        for entry in entries:
            parts.append(_format_entry(entry, db, include_params))
            parts.append("")

    user = "\n".join(parts)
    return system, user
