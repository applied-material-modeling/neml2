#!/usr/bin/env bash
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
