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
#include <cctype>
#include <ostream>

namespace neml2
{
class VariableName
{
public:
  VariableName() = default;
  VariableName(std::string value);
  VariableName(const char * value);

  operator std::string() const { return _value; }
  const std::string & str() const { return _value; }
  const char * c_str() const noexcept { return _value.c_str(); }

  friend bool operator==(const VariableName & a, const VariableName & b) noexcept;
  friend bool operator!=(const VariableName & a, const VariableName & b) noexcept;
  friend bool operator<(const VariableName & a, const VariableName & b) noexcept;
  friend std::ostream & operator<<(std::ostream & os, const VariableName & name);

private:
  static void validate(const std::string & s);

  static constexpr const char * invalid_c_str = "__invalid_variable_name__";
  std::string _value = invalid_c_str;
};

bool operator==(const VariableName & a, const VariableName & b) noexcept;
bool operator!=(const VariableName & a, const VariableName & b) noexcept;
bool operator<(const VariableName & a, const VariableName & b) noexcept;
std::ostream & operator<<(std::ostream & os, const VariableName & name);
std::stringstream & operator>>(std::stringstream & ss, VariableName & name);

/// User-defined literal for VariableName, e.g., "strain"_var
VariableName operator""_var(const char * str, size_t len);
} // namespace neml2
