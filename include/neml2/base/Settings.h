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

#pragma once

#include <string>
#include <vector>

namespace neml2
{
class OptionSet;

class Settings
{
public:
  static OptionSet expected_options();

  Settings(const OptionSet & options);

  ///@{
  /// Getters for the settings
  const std::string & buffer_name_separator() const { return _buffer_name_separator; }
  const std::string & parameter_name_separator() const { return _parameter_name_separator; }
  bool require_double_precision() const { return _require_double_precision; }
  const std::vector<std::string> & additional_libraries() const { return _additional_libraries; }
  bool disable_jit() const { return _disable_jit; }
  ///@}

private:
  /// Separator for buffer names
  const std::string _buffer_name_separator;

  /// Separator for parameter names
  const std::string _parameter_name_separator;

  /// Whether to enforce the use of double precision for floating point tensors
  const bool _require_double_precision;

  /// Additional dynamic libraries to load at runtime
  const std::vector<std::string> _additional_libraries;

  /// Disable JIT compilation
  const bool _disable_jit;
};
} // namespace neml2
