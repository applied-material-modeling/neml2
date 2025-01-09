// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <sstream>
#include <filesystem>
#include "utils.h"

int
try_hint(const std::string & stem, std::string & hint)
{
  namespace fs = std::filesystem;

  const auto hint_path = fs::path(hint);
  fs::path candidate;

  candidate = hint_path / stem;
  if (fs::is_directory(candidate))
  {
    hint = std::string(fs::absolute(candidate));
    return 0;
  }

  candidate = hint_path / "tests" / stem;
  if (fs::is_directory(candidate))
  {
    hint = std::string(fs::absolute(candidate));
    return 0;
  }

  candidate = hint_path / "share" / "neml2" / stem;
  if (fs::is_directory(candidate))
  {
    hint = std::string(fs::absolute(candidate));
    return 0;
  }

  return 1;
}

int
guess_test_dir(const std::string & stem, std::string & hint, const std::string & exec_prefix)
{
  namespace fs = std::filesystem;

  // Check if hint is an exact match
  const auto hint_path = fs::path(hint);
  if (fs::is_directory(hint_path) && hint_path.stem() == stem)
  {
    hint = std::string(fs::absolute(hint_path));
    return 0;
  }

  // Try alternative hints
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix;
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix + "/..";
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix + "/../..";
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix + "/../../..";
  if (try_hint(stem, hint) == 0)
    return 0;

  hint = exec_prefix + "/../../../..";
  if (try_hint(stem, hint) == 0)
    return 0;

  return 1;
}
