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

  // Settings is a special block that does not need a "type" field
  options.type() = "Settings";

  options.doc() = "Global settings for tensors, models, etc.";

  options.add<std::string>(
      "buffer_name_separator",
      "_",
      "Nested buffer name separator. The default is '_'. For example, a sub-model 'foo' which "
      "declares a buffer 'bar' will have a buffer named 'foo_bar'.");

  options.add<std::string>(
      "parameter_name_separator",
      "_",
      "Parameter name separator. The default is '_'. For example, a sub-model 'foo' which declares "
      "a parameter 'bar' will have a parameter named 'foo_bar'.");

  options.add<std::string>(
      "history_separator",
      "~",
      "History variable name separator. The default is '~'. For example, a variable 'x' will have "
      "its history variable for the previous time step named 'x~1', for the time step before that "
      "named 'x~2', etc.");

  options.add<std::string>(
      "rate_prefix",
      "",
      "Prefix for rate variables. The default is an empty string. For example, if the prefix is "
      "'rate_', then a variable named 'x' will have its rate variable named 'rate_x'.");
  options.add<std::string>(
      "rate_suffix",
      "_rate",
      "Suffix for rate variables. The default is '_rate'. For example, if the suffix is "
      "'_rate', then a variable named 'x' will have its rate variable named 'x_rate'.");

  options.add<std::string>(
      "residual_prefix",
      "",
      "Prefix for residual variables. The default is an empty string. For example, if the prefix "
      "is 'residual_', then a variable named 'x' will have its residual variable named "
      "'residual_x'.");
  options.add<std::string>(
      "residual_suffix",
      "_residual",
      "Suffix for residual variables. The default is '_residual'. For example, if the suffix is "
      "'_residual', then a variable named 'x' will have its residual variable named 'x_residual'.");

  options.add<bool>("require_double_precision",
                    true,
                    "Require double precision for all computations. An error will be thrown when "
                    "Model forward operators are called if the default dtype is not Float64. Set "
                    "this option to false to allow other precisions.");

  options.add<std::vector<std::string>>(
      "additional_libraries",
      {},
      "Additional dynamic libraries to load at runtime. The Registry from these libraries are "
      "merged into the current Registry singleton. This is required for using custom models "
      "defined in dynamic libraries not directly linked to libneml2. The paths are either absolute "
      "or relative to the current working directory.");

  options.add<bool>(
      "disable_jit",
      false,
      "Disable JIT compilation of models. This is useful for debugging or when the JIT compiler is "
      "not available. When set to false, each individual model can still selectively "
      "enable/disable JIT. When set to true, JIT is disabled globally, and it is an error to "
      "explicitly set jit to true for any model.");

  options.add<bool>(
      "linalg_solve_check_errors",
      false,
      "Whether to check for errors after solving linear systems. This is disabled by default for "
      "performance reasons, but it can be enabled to get better error messages when the solve "
      "fails.");

  return options;
}

Settings::Settings(const OptionSet & options)
  : _buffer_name_separator(options.get<std::string>("buffer_name_separator")),
    _parameter_name_separator(options.get<std::string>("parameter_name_separator")),
    _history_separator(options.get<std::string>("history_separator")),
    _rate_prefix(options.get<std::string>("rate_prefix")),
    _rate_suffix(options.get<std::string>("rate_suffix")),
    _residual_prefix(options.get<std::string>("residual_prefix")),
    _residual_suffix(options.get<std::string>("residual_suffix")),
    _require_double_precision(options.get<bool>("require_double_precision")),
    _additional_libraries(options.get<std::vector<std::string>>("additional_libraries")),
    _disable_jit(options.get<bool>("disable_jit")),
    _linalg_solve_check_errors(options.get<bool>("linalg_solve_check_errors"))
{
  // Load additional libraries which may contain custom objects
  for (const auto & lib : _additional_libraries)
    Registry::load(lib);
}
} // namespace neml2
