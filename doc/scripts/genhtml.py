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

import argparse
import shutil
import sys
import threading
from pathlib import Path

from preprocess import preprocess
from syntax_to_md import syntax_to_md
from utils import merge_files, quiet_run_and_log, render_file

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
if "site-packages" not in neml2.__path__[0]:
    logger.error("the imported neml2 package appears to be an editable installation.")
    logger.error("please install neml2 in non-editable mode before building the docs.")
    exit(1)


CONTENT_EXTS = {".md", ".py", ".cxx", ".h", ".txt", ".i"}
TUTORIAL_EXTS = {".out", ".svg", ".md", ".png"}
TUTORIAL_SKIP_DIRS = {"CMakeFiles"}


def sync_content(
    content_dir: Path, doc_dir: Path, build_dir: Path
) -> tuple[list[Path], list[Path]]:
    """
    Sync tracked source files from doc/content to build/content.

    Returns:
      - updated files (relative to doc/content), including non-markdown files
      - updated markdown .md.in files (absolute paths in build dir)
    """
    updated_files: list[Path] = []
    markdowns: list[Path] = []

    for ext in sorted(CONTENT_EXTS):
        for item in content_dir.rglob("*" + ext):
            rel_path = item.relative_to(doc_dir)
            dest_path = build_dir / rel_path
            if item.suffix == ".md":
                dest_path = dest_path.with_suffix(".md.in")

            if not dest_path.exists() or item.stat().st_mtime > dest_path.stat().st_mtime:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)
                updated_files.append(item.relative_to(content_dir))
                logger.trace("updated: {}", item.relative_to(content_dir))
                if dest_path.suffix == ".in":
                    markdowns.append(dest_path)

    return updated_files, markdowns


def sync_tutorial_outputs(tutorial_build_dir: Path, build_dir: Path):
    # Copy pre-generated tutorial outputs into build/content/tutorials.
    for ext in sorted(TUTORIAL_EXTS):
        for item in tutorial_build_dir.rglob("*" + ext):
            if any(part in TUTORIAL_SKIP_DIRS for part in item.parts):
                continue
            dest_path = build_dir / "content" / "tutorials" / item.relative_to(tutorial_build_dir)
            if not dest_path.exists() or item.stat().st_mtime > dest_path.stat().st_mtime:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)
                logger.trace("copied: {}", dest_path.relative_to(build_dir))


def extract_syntax(build_dir: Path):
    # Regenerate syntax reference markdown from neml2-syntax output.
    logger.info("")
    logger.info("extracting syntax...")
    syntax_yml = build_dir / "syntax.yml"
    success = quiet_run_and_log(["neml2-syntax", "--yaml", str(syntax_yml)])
    if not success or not syntax_yml.exists():
        logger.error("expected syntax file {} not found", syntax_yml)
        exit(1)

    missing = syntax_to_md(syntax_yml, build_dir / "content" / "syntax", build_dir / "syntax.err")
    if missing > 0:
        with open(build_dir / "syntax.err", "r") as log:
            content = log.read().strip()
            if content:
                logger.warning("syntax extraction reported the following issues:")
                for line in content.splitlines():
                    logger.warning(line)


def preprocess_markdowns(doxygen_layout: Path, markdowns: list[Path], build_content_dir: Path):
    # Preprocess only markdown inputs that changed in the latest sync.
    if not markdowns:
        logger.info("")
        logger.info("no markdown updates detected; skipping preprocess step")
        return
    logger.info("")
    logger.info("preprocessing {} markdown file(s)...", len(markdowns))
    preprocess(doxygen_layout, markdowns, build_content_dir)


def emit_doxygen_warnings(log_path: Path):
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "r") as f:
            for line in f:
                logger.warning(line.strip())


def generate_docs_with_doxygen(
    *,
    cfg_dir: Path,
    root_dir: Path,
    build_dir: Path,
    neml2_path: str,
    doxygen_path: str,
    message: str,
    doxyfile_name: str,
    templates: list[str],
    warning_log: str,
) -> bool:
    """Generate documentation by merging templates and invoking doxygen."""
    logger.info("")
    logger.info(message)
    doxyfile = build_dir / doxyfile_name
    merge_files([cfg_dir / name for name in templates], doxyfile)
    render_file(
        doxyfile,
        doxyfile,
        {
            "repo_root": root_dir.as_posix(),
            "build_dir": build_dir.as_posix(),
            "neml2_path": neml2_path,
        },
    )
    success = quiet_run_and_log([doxygen_path, "-q", str(doxyfile)])
    emit_doxygen_warnings(build_dir / warning_log)
    return success


def generate_cpp_docs(
    cfg_dir: Path, root_dir: Path, build_dir: Path, neml2_path: str, doxygen_path: str
) -> bool:
    # Build C++ API/reference HTML with doxygen.
    return generate_docs_with_doxygen(
        cfg_dir=cfg_dir,
        root_dir=root_dir,
        build_dir=build_dir,
        neml2_path=neml2_path,
        doxygen_path=doxygen_path,
        message="generating c++ documentation...",
        doxyfile_name="DoxyfileHTML",
        templates=["Doxyfile.in", "HTML.in"],
        warning_log="doxygen.html.log",
    )


def generate_python_docs(
    cfg_dir: Path, root_dir: Path, build_dir: Path, neml2_path: str, doxygen_path: str
) -> bool:
    # Build Python API/reference HTML with doxygen.
    return generate_docs_with_doxygen(
        cfg_dir=cfg_dir,
        root_dir=root_dir,
        build_dir=build_dir,
        neml2_path=neml2_path,
        doxygen_path=doxygen_path,
        message="generating python documentation...",
        doxyfile_name="DoxyfilePython",
        templates=["Doxyfile.in", "HTML.in", "Python.in"],
        warning_log="doxygen.python.log",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML documentation")
    parser.add_argument(
        "--log-level",
        "-L",
        type=str,
        default="INFO",
        help="Set the logging level. Ranking by severity, options are TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, and CRITICAL",
    )
    parser.add_argument("--build-dir", "-B", type=str, default="build/doc", help="Build directory")
    parser.add_argument(
        "--tutorial-build-dir",
        "-T",
        type=str,
        default="build/tutorials",
        help="Tutorial build directory",
    )
    parser.add_argument(
        "--clean",
        "-C",
        action="store_true",
        help="Clean the build directory before generating the docs",
    )
    parser.add_argument("--doxygen", type=str, default="doxygen", help="Path to doxygen executable")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve docs with livereload and rebuild when doc/content changes",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server host for --serve mode"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port for --serve mode")
    args = parser.parse_args()

    LiveReloadServer = None
    if args.serve:
        try:
            from livereload import Server as LiveReloadServer
        except ImportError:
            logger.error("livereload not found. Install it before using --serve.")
            exit(1)

    # Configure logging.
    logger.remove()
    logger.add(sys.stdout, format=msg_format, level=args.log_level.upper())
    logger.trace("")
    logger.trace("set log level to {}", args.log_level.upper())

    # Validate doxygen availability.
    success = quiet_run_and_log([args.doxygen, "--version"])
    if not success:
        logger.error("doxygen not found: {}", args.doxygen)
        logger.error("please install doxygen and make sure it is on your PATH")
        exit(1)

    # Ensure doxygen-awesome-css submodule is present.
    success = quiet_run_and_log(
        [
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--checkout",
            "contrib/doxygen-awesome-css",
        ]
    )
    if not success:
        logger.error("failed to initialize/update doxygen-awesome-css submodule")
        exit(1)

    # directories
    script_dir = Path(__file__).parent.resolve()
    doc_dir = script_dir.parent.resolve()
    content_dir = doc_dir / "content"
    root_dir = doc_dir.parent.resolve()
    cfg_dir = doc_dir / "config"
    build_dir = Path(args.build_dir).resolve()
    tutorial_build_dir = Path(args.tutorial_build_dir).resolve()
    doxygenlayout = cfg_dir / "DoxygenLayout.xml"
    logger.info("")
    logger.info("directories:")
    logger.info("      root: {}", root_dir)
    logger.info("       doc: {}", doc_dir)
    logger.info("   content: {}", content_dir)
    logger.info("  tutorial: {}", tutorial_build_dir)
    logger.info("     build: {}", build_dir)

    if not tutorial_build_dir.exists():
        logger.error("tutorial build directory not found: {}", tutorial_build_dir)
        logger.error("please build the tutorials before generating the docs by running examples.py")
        exit(1)
    if not doxygenlayout.exists():
        logger.error("DoxygenLayout.xml not found: {}", doxygenlayout)
        exit(1)

    # Prepare build directory.
    if args.clean and build_dir.exists():
        logger.info("")
        logger.info("cleaning build directory: {}...", build_dir)
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Sync documentation source content.
    logger.info("")
    logger.info("updating content...")
    updated_files, updated_markdowns = sync_content(content_dir, doc_dir, build_dir)
    logger.info("updated {} content file(s)", len(updated_files))

    # Sync tutorial outputs.
    logger.info("")
    logger.info("syncing tutorial output...")
    sync_tutorial_outputs(tutorial_build_dir, build_dir)

    # Generate syntax pages.
    extract_syntax(build_dir)

    # Preprocess updated markdown sources.
    preprocess_markdowns(doxygenlayout, updated_markdowns, build_dir / "content")

    # Generate C++ and Python documentation.
    success = generate_cpp_docs(cfg_dir, root_dir, build_dir, neml2.__path__[0], args.doxygen)
    if not success:
        logger.error("failed to generate c++ documentation")
        exit(1)
    success = generate_python_docs(cfg_dir, root_dir, build_dir, neml2.__path__[0], args.doxygen)
    if not success:
        logger.error("failed to generate python documentation")
        exit(1)

    if not args.serve:
        exit(0)

    serve_root = build_dir / "build" / "html"
    if not serve_root.exists():
        logger.error("serve root does not exist yet: {}", serve_root)
        exit(1)

    build_lock = threading.Lock()

    def rebuild_dynamic():
        # Avoid overlapping rebuilds when many file events fire together.
        if not build_lock.acquire(blocking=False):
            logger.warning("rebuild already in progress; skipping this event")
            return

        try:
            logger.info("")
            logger.info("change detected in content; updating docs...")
            updated_files, updated_markdowns = sync_content(content_dir, doc_dir, build_dir)
            if not updated_files:
                logger.trace("no tracked content updates found")
                return

            logger.info("content sync updated {} file(s)", len(updated_files))
            for path in updated_files:
                logger.trace("  {}", path.as_posix())

            preprocess_markdowns(doxygenlayout, updated_markdowns, build_dir / "content")

            ok_cpp = generate_cpp_docs(
                cfg_dir, root_dir, build_dir, neml2.__path__[0], args.doxygen
            )
            ok_py = generate_python_docs(
                cfg_dir, root_dir, build_dir, neml2.__path__[0], args.doxygen
            )
            if not (ok_cpp and ok_py):
                logger.warning("documentation rebuild completed with errors")
        finally:
            build_lock.release()

    logger.info("")
    logger.info("starting docs server with livereload...")
    logger.info("  root: {}", serve_root)
    logger.info("  url:  http://{}:{}/", args.host, args.port)
    logger.info("watching: {}", content_dir)

    if LiveReloadServer is None:
        logger.error("livereload server class is unavailable.")
        exit(1)

    server = LiveReloadServer()
    for ext in sorted(CONTENT_EXTS):
        server.watch((content_dir / f"**/*{ext}").as_posix(), rebuild_dynamic, delay=0.2)
    server.serve(root=serve_root.as_posix(), host=args.host, port=args.port)
