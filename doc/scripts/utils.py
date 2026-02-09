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

import subprocess
from loguru import logger
import shutil
from string import Template
from pathlib import Path


def pad_leading_space(line: list[str], space: int) -> list[str]:
    """Pads the leading space of a line by the given amount."""
    return [" " * space + l for l in line]


def quiet_run_and_log(command: list[str], **kwargs) -> bool:
    """Runs a command and logs the output."""
    logger.trace("running command: {}", " ".join(command))

    # check if the command exists
    path = shutil.which(command[0])
    if path is None:
        logger.trace(f"{command[0]} not found in PATH")
        return False

    # run the command and log the output
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode == 0:
        logger.trace("command succeeded")
    else:
        logger.trace("command failed with return code {}: {}", result.returncode, " ".join(command))
        logger.trace("stdout:")
        for line in result.stdout.splitlines():
            logger.trace("  {}", line)
        logger.trace("stderr:")
        for line in result.stderr.splitlines():
            logger.trace("  {}", line)
    return result.returncode == 0


def merge_files(files: list[Path], dest: Path):
    with open(dest, "w") as out:
        for file in files:
            with open(file, "r") as f:
                out.write(f.read())
                out.write("\n")


def render_file(template_path: Path, output_path: Path, variables: dict[str, str]):
    text = Path(template_path).read_text()
    tpl = Template(text)

    try:
        rendered = tpl.substitute(variables)
    except KeyError as e:
        logger.error(f"missing variable for substitution: {e}")
        return

    Path(output_path).write_text(rendered)
