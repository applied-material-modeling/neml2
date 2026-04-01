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

"""
neml2.reader
============

Tools for parsing NEML2 input files and generating natural-language
explanations using an LLM.

Quick start::

    from neml2.reader import explain

    # Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in environment
    text = explain("path/to/model.i", provider="anthropic")
    print(text)

For finer control (inspect the prompt without calling an LLM)::

    from neml2.reader import describe

    system, user = describe("path/to/model.i", syntax_db="build/doc/syntax.yml")
    print(user)
"""

from pathlib import Path
from typing import Optional, Union

from ._parser import ModelEntry, parse_input
from ._syntax import SyntaxDB
from ._prompt import build_prompt
from ._llm import LLMClient


def _resolve_db(syntax_db) -> SyntaxDB:
    if isinstance(syntax_db, SyntaxDB):
        return syntax_db
    return SyntaxDB(syntax_path=syntax_db)


def describe(
    input_file: Union[str, Path],
    syntax_db: Union[Path, SyntaxDB],
    include_params: bool = True,
) -> tuple:
    """
    Parse a NEML2 input file and build an LLM-ready prompt, without calling
    any LLM.

    This is the lower-level API, useful for inspecting the prompt, testing,
    or feeding into your own LLM client.

    Args:
        input_file: Path to the HIT input file.
        syntax_db: Path to ``syntax.yml`` or a :class:`SyntaxDB` instance.
        sections: Limit to these top-level section names (e.g.
            ``["Models"]``). Defaults to all sections.
        include_params: Whether to include per-parameter details in the
            prompt.

    Returns:
        A ``(system_prompt, user_prompt)`` tuple of strings.

    Raises:
        FileNotFoundError: If ``input_file`` does not exist.
        ValueError: If the input file has invalid HIT syntax.
    """
    parsed = parse_input(input_file)
    db = _resolve_db(syntax_db)
    return build_prompt(parsed, db, sections=["Models"], include_params=include_params)


def explain(
    input_file: Union[str, Path],
    syntax_db: Union[Path, SyntaxDB],
    client: LLMClient,
    include_params: bool = True,
    system_context: Optional[str] = None,
) -> str:
    """
    Parse a NEML2 input file and return an LLM-generated natural-language
    explanation.

    Args:
        input_file: Path to the HIT input file.
        syntax_db: Path to ``syntax.yml`` or a :class:`SyntaxDB` instance.
        client: An instance of a class that implements the :class:`LLMClient` protocol.
        include_params: Whether to include per-parameter details in the
            prompt.
        system_context: Optional extra context appended to the system prompt
            (e.g. ``"The user is a structural engineer"``).

    Returns:
        The LLM's natural-language explanation as a string.

    Raises:
        FileNotFoundError: If ``input_file`` does not exist.
        ValueError: If the input file has invalid HIT syntax or the provider
            is unrecognised.
        ImportError: If the required package for the chosen provider is not
            installed.
    """
    parsed = parse_input(input_file)
    db = _resolve_db(syntax_db)
    system, user = build_prompt(
        parsed,
        db,
        sections=["Models"],
        include_params=include_params,
        system_context=system_context,
    )
    return client.complete(system, user)


__all__ = [
    "explain",
    "describe",
    "parse_input",
    "ModelEntry",
    "SyntaxDB",
    "LLMClient",
    "build_prompt",
]
