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
set -euo pipefail

if command -v clang-format >/dev/null 2>&1; then
  command -v clang-format
  exit 0
fi

if command -v clang-format-20 >/dev/null 2>&1; then
  command -v clang-format-20
  exit 0
fi

if command -v clang-format-19 >/dev/null 2>&1; then
  command -v clang-format-19
  exit 0
fi

if command -v brew >/dev/null 2>&1; then
  llvm_prefix="$(brew --prefix llvm 2>/dev/null || true)"
  if [ -n "${llvm_prefix}" ] && [ -x "${llvm_prefix}/bin/clang-format" ]; then
    printf '%s\n' "${llvm_prefix}/bin/clang-format"
    exit 0
  fi
fi

if [ -x /opt/homebrew/opt/llvm/bin/clang-format ]; then
  printf '%s\n' /opt/homebrew/opt/llvm/bin/clang-format
  exit 0
fi

if [ -x /usr/local/opt/llvm/bin/clang-format ]; then
  printf '%s\n' /usr/local/opt/llvm/bin/clang-format
  exit 0
fi

exit 1
