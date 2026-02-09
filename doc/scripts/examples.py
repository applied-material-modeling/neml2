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
import argparse
import os
from pathlib import Path
import shutil
from utils import quiet_run_and_log
import subprocess
import concurrent.futures
import nbformat

try:
    from loguru import logger
except ImportError:
    print("loguru not found. Please install loguru before building the docs.")
    exit(1)

msg_format = "[<green>{elapsed}</green>] <cyan>{file.name: <17}</cyan>| <level>{message}</level>"
logger.remove()
logger.add(sys.stdout, format=msg_format, level="INFO")

logger.info("checking for neml2 package installation...")
try:
    import neml2
except ImportError:
    logger.error("neml2 not found. Please install neml2 before building the docs.")
    logger.info("python executable: {}", sys.executable)
    exit(1)
logger.success("")
logger.success("neml2 package:")
logger.success("  path:    {}", neml2.__path__[0])
logger.success("  version: {}", neml2.__version__)
logger.success("  hash:    {}", neml2.__hash__)

logger.info("")
logger.info("checking if the imported neml2 package is editable...")
if not "site-packages" in neml2.__path__[0]:
    logger.error("the imported neml2 package appears to be an editable installation.")
    logger.error("please install neml2 in non-editable mode before building the docs.")
    exit(1)


def run_example(path: Path, cmd_gen) -> tuple[bool, str]:
    msg = "\n"
    msg += "running example: {}\n".format(path.name)
    msg += "changing directory to: {}\n".format(path.parent)
    cmd = cmd_gen(path)
    result = subprocess.run(cmd, cwd=path.parent, capture_output=True, text=True)
    if result.returncode == 0:
        # write stdout to log file
        log_file = path.with_suffix(".out")
        msg += "writing output to: {}\n".format(log_file)
        with open(log_file, "w") as f:
            f.write(result.stdout)
    else:
        msg += "error running example: {}\n".format(path.relative_to(build_dir))
        msg += "command failed with return code {}: {}\n".format(result.returncode, " ".join(cmd))
        msg += "stdout:\n"
        for line in result.stdout.splitlines():
            msg += "  {}\n".format(line)
        msg += "stderr:\n"
        for line in result.stderr.splitlines():
            msg += "  {}\n".format(line)
        return False, msg
    return True, msg


def run_examples_in_pool(examples: list[Path], cmd_gen, max_workers: int) -> tuple[int, int]:
    nfail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_example, example, cmd_gen): example for example in examples}
        for future in concurrent.futures.as_completed(futures):
            success, msg = future.result()
            if success:
                for line in msg.splitlines():
                    logger.info(line)
            else:
                for line in msg.splitlines():
                    logger.warning(line)
                nfail += 1
    return nfail, len(examples)


def check_nb_execution(nb_path: Path) -> list[int]:
    """
    Get the number of unexecuted code cells in a jupyter notebook.
    This is used to check if a notebook has been executed before converting it to markdown for doxygen.
    """
    nb = nbformat.read(nb_path, as_version=4)
    unexecuted = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and cell.execution_count is None:
            unexecuted.append(i)
    return unexecuted


def append_page_ref(md_path: Path, ref: str):
    """Append a page reference to the end of a markdown file for use in doxygen."""
    with md_path.open("r") as f:
        first = f.readline()
        rest = f.read()

        if not first.startswith("#"):
            logger.warning("notebook markdown file does not start with a header: {}", md_path)
            return

        md_path.write_text(first.rstrip() + " {#" + ref + "}\n" + rest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and run examples")
    parser.add_argument(
        "--log-level",
        "-L",
        type=str,
        default="INFO",
        help="Set the logging level. Ranking by severity, options are TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, and CRITICAL",
    )
    parser.add_argument(
        "--build-dir", "-B", type=str, default="build/tutorials", help="Build directory"
    )
    parser.add_argument(
        "--clean",
        "-C",
        action="store_true",
        help="Clean the build directory before building the docs",
    )

    parser.add_argument("--skip-cpp", action="store_true", help="Skip C++ examples")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python examples")
    parser.add_argument("--skip-ipynb", action="store_true", help="Skip Jupyter notebook examples")

    parser.add_argument(
        "--cmake-path",
        type=str,
        default="cmake",
        help="Path to the cmake executable",
    )
    parser.add_argument(
        "--cmake-configure-args",
        type=str,
        default="",
        help="Additional arguments to pass to cmake during configuration",
    )
    parser.add_argument(
        "--cmake-build-args",
        type=str,
        default="",
        help="Additional arguments to pass to cmake during build",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=0,
        help="Maximum number of examples to run in parallel. Default is min(8, number of CPU cores)",
    )

    args = parser.parse_args()

    # set log level
    logger.remove()
    logger.add(sys.stdout, format=msg_format, level=args.log_level.upper())
    logger.trace("")
    logger.trace("set log level to {}", args.log_level.upper())

    # directories
    script_dir = Path(__file__).parent
    doc_dir = script_dir.parent.resolve()
    root_dir = doc_dir.parent.resolve()
    build_dir = Path(args.build_dir).resolve()
    tutorial_dir = doc_dir / "content" / "tutorials"
    logger.info("")
    logger.info("directories:")
    logger.info("      root: {}", root_dir)
    logger.info("  tutorial: {}", tutorial_dir)
    logger.info("     build: {}", build_dir)

    # create/clean build directory
    if args.clean and build_dir.exists():
        logger.info("")
        logger.info("cleaning build directory: {}...", build_dir)
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # copy content from doc/content/tutorials to the build directory
    # only copy files with the following extensions: .py, .i, .pt
    # only copy if the source is newer than the destination
    logger.info("")
    logger.info("updating content...")
    exts = {".py", ".i", ".pt"}
    for ext in exts:
        for item in tutorial_dir.rglob("*" + ext):
            rel_path = item.relative_to(tutorial_dir)
            dest_path = build_dir / rel_path
            if item.is_dir():
                dest_path.mkdir(parents=True, exist_ok=True)
            else:
                if not dest_path.exists() or item.stat().st_mtime > dest_path.stat().st_mtime:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    logger.trace("updated: {}", rel_path)

    max_workers = args.jobs if args.jobs > 0 else min(8, (os.cpu_count() or 1))
    logger.trace("using {} worker threads for examples", max_workers)

    if not args.skip_cpp:
        # check if cmake exists
        logger.trace("")
        logger.trace("checking for cmake...", args.cmake_path)
        success = quiet_run_and_log([args.cmake_path, "--version"])
        if not success:
            logger.error("cmake not found: {}", args.cmake_path)
            exit(1)

        # configure c++ tutorials
        logger.info("")
        logger.info("configuring c++ tutorials...")
        command = [
            args.cmake_path,
            "-Dneml2_ROOT={}".format(neml2.__path__[0]),
            "-S",
            str(tutorial_dir),
            "-B",
            str(build_dir),
        ] + args.cmake_configure_args.split()
        logger.trace("command: {}", " ".join(command))
        success = quiet_run_and_log(command)
        if not success:
            logger.error("cmake configuration failed")
            exit(1)

        # build c++ tutorials
        logger.info("")
        logger.info("building c++ tutorials...")
        command = [args.cmake_path, "--build", str(build_dir)] + args.cmake_build_args.split()
        logger.trace("command: {}", " ".join(command))
        result = subprocess.run(command)
        if result.returncode != 0:
            logger.error("cmake build failed")
            exit(1)

        # run c++ examples
        logger.info("")
        logger.info("running c++ examples...")
        example_list = build_dir / "examples.txt"
        if not example_list.exists():
            logger.error("expected file not found: {}", example_list)
            exit(1)
        with open(example_list) as f:
            examples = f.readlines()
        # sort examples so that the behavior is deterministic
        examples = [Path(t.strip()).resolve() for t in examples]
        examples.sort()
        nfail, total = run_examples_in_pool(examples, lambda p: [str(p)], max_workers)
        if nfail > 0:
            logger.error("")
            logger.error("{}/{} examples failed", nfail, total)
            exit(1)
        else:
            logger.success("")
            logger.success("successfully ran all {} c++ examples", total)

    if not args.skip_python:
        # run python examples
        logger.info("")
        logger.info("running python examples...")
        # sort tutorials so that the behavior is deterministic
        examples = list(build_dir.rglob("*.py"))
        examples.sort()
        nfail, total = run_examples_in_pool(
            examples, lambda p: [sys.executable, str(p)], max_workers
        )
        if nfail > 0:
            logger.error("")
            logger.error("{}/{} python examples failed", nfail, total)
            exit(1)
        else:
            logger.success("")
            logger.success("successfully ran all {} python examples", total)

    if not args.skip_ipynb:
        #         jupyter nbconvert notebook.ipynb \
        #   --to markdown \
        #   --output notebook.md \
        #   --ExtractOutputPreprocessor.enabled=True \
        #   --TemplateExporter.exclude_input_prompt=True \
        #   --TemplateExporter.exclude_output_prompt=True
        # run jupyter notebook examples
        logger.info("")
        logger.info("converting jupyter notebooks to markdowns...")
        # sort tutorials so that the behavior is deterministic
        nbs = list(root_dir.rglob("*.ipynb"))
        nbs.sort()
        nfail = 0
        for nb in nbs:
            logger.info("")
            logger.info("converting notebook: {}", nb.relative_to(root_dir))
            unexecuted_cells = check_nb_execution(nb)
            if len(unexecuted_cells) > 0:
                logger.error(
                    "notebook has {} unexecuted code cells: {}",
                    len(unexecuted_cells),
                    unexecuted_cells,
                )
                exit(1)
            md = build_dir / nb.relative_to(root_dir).with_suffix(".md")
            command = [
                "jupyter",
                "nbconvert",
                str(nb),
                "--to",
                "markdown",
                "--output",
                str(md),
                "--ExtractOutputPreprocessor.enabled=True",
                "--TemplateExporter.exclude_input_prompt=True",
                "--TemplateExporter.exclude_output_prompt=True",
            ]
            success = quiet_run_and_log(command)
            if not success:
                logger.error("error converting notebook: {}", nb.relative_to(root_dir))
                nfail += 1
            # assign a reliable page reference to the markdown file for use in doxygen
            ref = (
                str(nb.relative_to(root_dir))
                .replace(os.sep, "/")
                .replace("/", "-")
                .replace(".ipynb", "")
            )
            append_page_ref(md, ref)
        if nfail > 0:
            logger.error("")
            logger.error("{}/{} jupyter notebook examples failed", nfail, len(nbs))
            exit(1)
        else:
            logger.success("")
            logger.success("successfully ran all {} jupyter notebook examples", len(nbs))
