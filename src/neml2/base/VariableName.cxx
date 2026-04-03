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

#include <cctype>
#include <sstream>

#include "neml2/base/VariableName.h"
#include "neml2/misc/errors.h"

namespace neml2
{
VariableName::VariableName(std::string value)
  : _value(std::move(value))
{
  validate(_value);
}

VariableName::VariableName(const char * value)
  : _value(value)
{
  validate(_value);
}

void
VariableName::validate(const std::string & s)
{
  if (s.empty())
    throw NEMLException("Variable name cannot be empty");

  // Cannot be the invalid name
  if (s == invalid_c_str)
    throw NEMLException("Variable name cannot be '" + std::string(invalid_c_str) +
                        "' since it is reserved as the invalid name");

  for (char ch : s)
  {
    const unsigned char uch = static_cast<unsigned char>(ch);

    // Do not allow:
    // - whitespace
    // - comma
    // - semicolon
    // - dot
    if (std::isspace(uch))
      throw NEMLException("Variable name '" + s + "' contains white space, which is not allowed");
    if (uch == ',' || uch == ';' || uch == '.')
      throw NEMLException("Variable name '" + s + "' contains character '" + std::string(1, ch) +
                          "', which is not allowed");
  }

  // First character cannot be a digit
  if (std::isdigit(static_cast<unsigned char>(s.front())))
    throw NEMLException("Variable name '" + s + "' cannot start with a digit");
}

bool
operator==(const VariableName & a, const VariableName & b) noexcept
{
  return a._value == b._value;
}

bool
operator!=(const VariableName & a, const VariableName & b) noexcept
{
  return !(a == b);
}

bool
operator<(const VariableName & a, const VariableName & b) noexcept
{
  return a._value < b._value;
}

std::ostream &
operator<<(std::ostream & os, const VariableName & name)
{
  return os << name._value;
}

std::stringstream &
operator>>(std::stringstream & ss, VariableName & name)
{
  std::string str;
  ss >> str;
  name = VariableName(str);
  return ss;
}

VariableName
operator""_var(const char * str, size_t len)
{
  return VariableName(std::string(str, len));
}

} // namespace neml2
