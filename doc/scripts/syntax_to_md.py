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

import yaml
import sys
import re
from pathlib import Path
from loguru import logger
import unicodedata as ud


def remove_namespace(type):
    type = type.replace("std::", "")
    type = type.replace("torch::", "")
    type = type.replace("c10::", "")
    type = type.replace("at::", "")
    type = type.replace("neml2::", "")
    type = type.replace("utils::", "")
    type = type.replace("jit::", "")
    return type


def normalize_string_types(type):
    return re.sub(
        r"(?:\b[\w:]+::)?basic_string<\s*char\s*,\s*(?:[\w:]+::)?char_traits<char>\s*,\s*(?:[\w:]+::)?allocator<char>\s*>\s*",
        "std::string",
        type,
    )


def normalize_vector_types(type):
    return re.sub(
        r"(?:\b[\w:]+::)?vector<\s*([^,>]+)\s*,\s*(?:[\w:]+::)?allocator<[^>]+>\s*>\s*",
        r"vector<\1>",
        type,
    )


def demangle(type):
    type = re.sub(r"SmallVector<[^>]*>", "tensor shape", type)
    type = normalize_string_types(type)
    type = normalize_vector_types(type)
    type = re.sub(r",\s*(?:[\w:]+::)?allocator<[^>]+>\s*", "", type)
    type = remove_namespace(type)
    type = type.replace("LabeledAxisAccessor", "variable name")
    type = re.sub("TensorName<(.+)>", r"\1 🔗", type)
    type = re.sub("vector<(.+)>", r"list of \1", type)
    # Call all integral/floating point types "number", as this syntax documentation faces the general audience potentially without computer science background
    type = type.replace("int", "number")
    type = type.replace("long", "number")
    type = type.replace("double", "number")
    type = type.replace("unsigned", "non-negative")

    return type


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
Clicking on the option with a triangle bullet ▸ next to it will expand/collapse its detailed information.

\\note
Type name written in PascalCase typically refer to a NEML2 object type, oftentimes a primitive tensor type.

\\note
The 🔗 symbol means that the tensor value can be cross-reference another object. See @ref tutorials-models-model-parameters-revisited for details.

\\note
You can always use `Ctrl`+`F` or `Cmd`+`F` to search the entire page.

"""
    if section == "Models":
        prologue += """The following symbols are used throughout the documentation to denote different components of function definition.
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
    Convert the syntax YAML file extracted by neml2-syntax into markdown files for documentation.

    Args:
        syntax_file: the input YAML file containing the syntax information
        outdir: the output directory to write the markdown files to
        logfile: the output log file to write any syntax issues to

    Returns:
        the number of syntax issues found, where an issue can be either a missing section, missing object description,
        or missing option description. Note that syntax issues do not necessarily indicate a problem with the
        code, but they do indicate missing documentation that should be filled in for better user experience.
    """

    with open(syntax_file, "r") as stream:
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
                    "The first non-printable character is at line {line}, column {col} ({codepoint}, {category}, {name})".format(
                        **np
                    )
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
                    "Refer to [System Documentation](@ref system-{}) for detailed explanation about this system.\n\n".format(
                        section.lower()
                    )
                )
                for type, params in syntax.items():
                    if params["section"] != section:
                        continue
                    input_type = remove_namespace(params["type"]["value"])
                    stream.write("### {} {{#{}}}\n\n".format(input_type, input_type.lower()))
                    if params["doc"]:
                        stream.write("{}\n".format(params["doc"]))
                    else:
                        missing += 1

                        log.write(
                            "  * '{}/{}' is missing object description\n".format(
                                section, input_type
                            )
                        )
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

                        param_type = demangle(info["type"])
                        param_value = postprocess(info["value"], param_type)
                        stream.write("<details>\n")
                        if not info["doc"]:
                            stream.write("  <summary>`{}`</summary>\n\n".format(param_name))
                            missing += 1
                            log.write(
                                "  * '{}/{}/{}' is missing option description\n".format(
                                    section, input_type, param_name
                                )
                            )
                        else:

                            stream.write(
                                "  <summary>`{}` {} {}</summary>\n\n".format(
                                    param_name, ftype_icon(info["ftype"]), info["doc"]
                                )
                            )
                            if "\\f" in info["doc"]:
                                log.write(
                                    "  * '{}/{}/{}' has formula in its option description\n".format(
                                        section, input_type, param_name
                                    )
                                )
                        stream.write("  - <u>Type</u>: {}\n".format(param_type))
                        if param_value:
                            stream.write("  - <u>Default</u>: {}\n".format(param_value))
                        stream.write("</details>\n")
                    stream.write("\n")
                    stream.write("Detailed documentation [link](@ref {})\n\n".format(type))

        if missing == 0:
            log.write("No syntax error, good job! :purple_heart:")
        else:
            print("*" * 79, file=sys.stderr)
            print("Syntax errors have been written to", logfile, file=sys.stderr)
            print("*" * 79, file=sys.stderr)

        return missing
