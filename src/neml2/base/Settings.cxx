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

#include "neml2/base/Settings.h"
#include "neml2/base/OptionSet.h"
#include "neml2/base/Registry.h"

namespace neml2
{
OptionSet
Settings::expected_options()
{
  OptionSet options;
  options.section() = "Settings";
  options.doc() = "Global settings for tensors, models, etc.";

  options.set<std::string>("type") = "Settings";

  options.set<std::string>("buffer_name_separator") = "_";
  options.set("buffer_name_separator").doc() = "Nested buffer name separator. The default is '_'. "
                                               "For example, a sub-model 'foo' which declares "
                                               "a buffer 'bar' will have a buffer named 'foo_bar'.";

  options.set<std::string>("parameter_name_separator") = "_";
  options.set("parameter_name_separator").doc() =
      "Parameter name separator. The default is '_'. For example, a sub-model 'foo' which declares "
      "a parameter 'bar' will have a parameter named 'foo_bar'.";

  options.set<bool>("require_double_precision") = true;
  options.set("require_double_precision").doc() =
      "Require double precision for all computations. An error will be thrown when Model forward "
      "operators are called if the default dtype is not Float64. Set this option to false to allow "
      "other precisions.";

  options.set<std::vector<std::string>>("additional_libraries");
  options.set("additional_libraries").doc() =
      "Additional dynamic libraries to load at runtime. The Registry from these libraries are "
      "merged into the current Registry singleton. This is required for using custom models "
      "defined in dynamic libraries not directly linked to libneml2. The paths are either absolute "
      "or relative to the current working directory.";

  options.set<bool>("disable_jit") = false;
  options.set("disable_jit").doc() =
      "Disable JIT compilation of models. This is useful for debugging or when the JIT compiler is "
      "not available. When set to false, each individual model can still selectively "
      "enable/disable JIT. When set to true, JIT is disabled globally, and it is an error to "
      "explicitly set jit to true for any model.";

  return options;
}

Settings::Settings(const OptionSet & options)
  : _buffer_name_separator(options.get<std::string>("buffer_name_separator")),
    _parameter_name_separator(options.get<std::string>("parameter_name_separator")),
    _require_double_precision(options.get<bool>("require_double_precision")),
    _additional_libraries(options.get<std::vector<std::string>>("additional_libraries")),
    _disable_jit(options.get<bool>("disable_jit"))
{
  // Load additional libraries which may contain custom objects
  for (const auto & lib : _additional_libraries)
    Registry::load(lib);
}
} // namespace neml2
