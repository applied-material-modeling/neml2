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
import unicodedata as ud
from pathlib import Path

import yaml
from loguru import logger


def postprocess(value, type):
    if type == "bool":
        value = "true" if value else "false"
    return value


def get_sections(syntax):
    sections = [params["section"] for type, params in syntax.items()]
    return list(dict.fromkeys(sections))


def ftype_icon(ftype):
    if ftype == "INPUT":
        return "🇮"
    elif ftype == "OUTPUT":
        return "🇴"
    elif ftype == "PARAMETER":
        return "🇵"
    elif ftype == "BUFFER":
        return "🇧"

    return ""


def section_prologue(section):
    prologue = """\\note
Clicking on the option with a triangle bullet ▸ next to it will
expand/collapse its detailed information.

\\note
Type name written in PascalCase typically refer to a NEML2 object type,
oftentimes a primitive tensor type.

\\note
The 🔗 symbol means that the tensor value can be cross-reference another
object. See @ref tutorials-models-model-parameters-revisited for details.

\\note
You can always use `Ctrl`+`F` or `Cmd`+`F` to search the entire page.

"""
    if section == "Models":
        prologue += """The following symbols are used throughout the documentation to denote
different components of function definition.
- 🇮: input variable
- 🇴: output variable
- 🇵: parameter
- 🇧: buffer
"""

    return prologue


def first_nonprintable(path: Path, encoding="utf-8") -> dict | None:
    BAD_CATEGORIES = {"Cc", "Cf", "Cs"}
    ALLOW = {"\n", "\r", "\t"}
    text = path.read_text(encoding=encoding, errors="surrogateescape")

    line = 1
    col = 1
    for idx, ch in enumerate(text):
        if ch == "\n":
            line += 1
            col = 1
            continue

        cat = ud.category(ch)
        if ch not in ALLOW and cat in BAD_CATEGORIES:
            return {
                "line": line,
                "col": col,
                "index": idx,
                "char": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "category": cat,
                "name": ud.name(ch, "<no name>"),
            }

        col += 1

    return None


def syntax_to_md(syntax_file: Path, outdir: Path, logfile: Path) -> int:
    """
    Convert the syntax YAML file extracted by neml2-syntax into markdown files
    for documentation.

    Args:
        syntax_file: the input YAML file containing the syntax information
        outdir: the output directory to write the markdown files to
        logfile: the output log file to write any syntax issues to

    Returns:
        the number of syntax issues found, where an issue can be either a
        missing section, missing object description, or missing option
        description. Note that syntax issues do not necessarily indicate a
        problem with the code, but they do indicate missing documentation that
        should be filled in for better user experience.
    """

    with open(syntax_file) as stream:
        try:
            syntax = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            logger.error(f"Error reading YAML file: {syntax_file}")
            err_msg = str(e)
            for line in err_msg.splitlines():
                logger.error(f"  {line}")
            np = first_nonprintable(syntax_file)
            if np is not None:
                logger.error(
                    "The first non-printable character is at line {line}, "
                    "column {col} ({codepoint}, {category}, {name})".format(**np)
                )
            exit(1)
    outdir.mkdir(parents=True, exist_ok=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)

    with open(logfile, "w") as log:
        missing = 0
        log.write("### Syntax check\n\n")
        sections = get_sections(syntax)
        for section in sections:
            if not section:
                missing += 1
                log.write(
                    "Section is not defined for one of the objects, did you forget to "
                    "set options.section() in one of its base classes?"
                )
                continue
            with open((outdir / section.lower()).with_suffix(".md"), "w") as stream:
                stream.write("# [{}] {{#{}}}\n\n".format(section, "syntax-" + section.lower()))
                stream.write("[TOC]\n\n")
                stream.write(section_prologue(section))
                stream.write("\n")
                stream.write("## Available objects and their input file syntax\n\n")
                stream.write(
                    f"Refer to [System Documentation](@ref system-{section.lower()})"
                    " for detailed explanation about this system.\n\n"
                )
                for type, params in syntax.items():
                    if params["section"] != section:
                        continue
                    stream.write(f"### {type} {{#{type.lower()}}}\n\n")
                    if params["doc"]:
                        stream.write("{}\n".format(params["doc"]))
                    else:
                        missing += 1

                        log.write(f"  * '{section}/{type}' is missing object description\n")
                    for param_name, info in params.items():
                        if param_name == "section":
                            continue
                        if param_name == "doc":
                            continue
                        if param_name == "name":
                            continue
                        if param_name == "type":
                            continue
                        if info["suppressed"]:
                            continue

                        param_type = info["type"]
                        param_value = postprocess(info["value"], param_type)
                        stream.write("<details>\n")
                        if not info["doc"]:
                            stream.write(f"  <summary>`{param_name}`</summary>\n\n")
                            missing += 1
                            log.write(
                                f"  * '{section}/{type}/{param_name}' "
                                "is missing option description\n"
                            )
                        else:
                            stream.write(
                                "  <summary>`{}` {} {}</summary>\n\n".format(
                                    param_name, ftype_icon(info["ftype"]), info["doc"]
                                )
                            )
                            if "\\f" in info["doc"]:
                                log.write(
                                    f"  * '{section}/{type}/{param_name}' "
                                    "has formula in its option description\n"
                                )
                        stream.write(f"  - <u>Type</u>: {param_type}\n")
                        stream.write(
                            "  - <u>Required</u>: {}\n".format("Yes" if info["required"] else "No")
                        )
                        if param_value:
                            stream.write(f"  - <u>Default</u>: {param_value}\n")
                        stream.write("</details>\n")
                    stream.write("\n")
                    stream.write(f"Detailed documentation [link](@ref {type.lower()})\n\n")

        if missing == 0:
            log.write("No syntax error, good job! :purple_heart:")
        else:
            print("*" * 79, file=sys.stderr)
            print("Syntax errors have been written to", logfile, file=sys.stderr)
            print("*" * 79, file=sys.stderr)

        return missing
