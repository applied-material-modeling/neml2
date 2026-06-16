#!/usr/bin/env bash

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

# Build (and optionally serve) the sphinx HTML docs.
#
# Default invocation matches the CLAUDE.md canonical command: parallel
# build, warnings-as-errors, --keep-going, output to doc/_build/html.
# Pre-creates the .jupyter_cache directory before invoking sphinx-build
# because myst-nb's jupyter-cache backend races to mkdir it across
# parallel workers; one worker wins and the rest fail with FileExistsError.
# Pre-creating is the simplest workaround and lets us keep -j auto.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
DOC_DIR=$(dirname "$SCRIPT_DIR")
REPO_DIR=$(dirname "$DOC_DIR")

DEST="$DOC_DIR/_build/html"
# myst-nb's default jupyter-cache path is sibling to the sphinx outdir
# (i.e. one level up + ``/.jupyter_cache``). With ``$DEST=doc/_build/html``
# that puts the cache at ``doc/_build/.jupyter_cache``. The script
# tracks this explicitly so ``--clean`` actually wipes the cache and
# the pre-create races the right directory.
CACHE_DIR="$(dirname "$DEST")/.jupyter_cache"
JOBS="auto"
PORT=8765
CLEAN=0
SERVE=0
STRICT=1  # -W (warnings as errors)

usage() {
  cat <<EOF
Usage: $0 [options]

Build the sphinx HTML docs under doc/ and optionally serve them locally.

Options:
  --clean             Remove the destination directory before building.
  --serve             After building, start an HTTP server on the destination.
  --dest PATH         Output directory (default: $DEST, relative to repo root).
  --port N            Port to serve on (default: $PORT). Bound to 127.0.0.1.
  -j, --jobs N        Parallel workers for sphinx-build (default: $JOBS).
  --no-strict         Do not treat warnings as errors (drop sphinx-build -W).
  -h, --help          Print this help.

The serve mode binds to 127.0.0.1 only; reach it from another machine via
an SSH tunnel: ssh -L PORT:127.0.0.1:PORT user@host
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --clean)      CLEAN=1; shift;;
    --serve)      SERVE=1; shift;;
    --dest)       DEST="$2"; shift 2;;
    --port)       PORT="$2"; shift 2;;
    -j|--jobs)    JOBS="$2"; shift 2;;
    --no-strict)  STRICT=0; shift;;
    -h|--help)    usage; exit 0;;
    *)            echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

cd "$REPO_DIR"

if [ "$CLEAN" -eq 1 ]; then
  echo "Cleaning: $DEST $CACHE_DIR"
  rm -rf "$DEST" "$CACHE_DIR"
fi

# Pre-create the jupyter cache dir so -j auto workers don't race the mkdir
# (myst-nb's jupyter-cache backend hits FileExistsError otherwise).
mkdir -p "$CACHE_DIR"

SPHINX_ARGS=(-j "$JOBS" --keep-going -b html)
if [ "$STRICT" -eq 1 ]; then
  SPHINX_ARGS+=(-W)
fi
SPHINX_ARGS+=("$DOC_DIR" "$DEST")

echo "Running: sphinx-build ${SPHINX_ARGS[*]}"
sphinx-build "${SPHINX_ARGS[@]}"

if [ "$SERVE" -eq 1 ]; then
  HOST=$(hostname)
  echo
  echo "Serving $DEST on http://127.0.0.1:$PORT/"
  echo "Tunnel from your local machine with:"
  echo "  ssh -L $PORT:127.0.0.1:$PORT $USER@$HOST"
  echo "Then browse to http://127.0.0.1:$PORT/ locally. Ctrl-C to stop."
  echo
  exec python -m http.server "$PORT" --bind 127.0.0.1 --directory "$DEST"
fi
