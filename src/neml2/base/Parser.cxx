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

#include <iostream>

#include "neml2/base/Parser.h"
#include "neml2/base/LabeledAxisAccessor.h"

#include <iostream>

namespace neml2
{
const std::vector<std::string> Parser::sections = {
    "Tensors", "Solvers", "Data", "Models", "Drivers", "Schedulers"};

namespace utils
{
template <>
bool
parse_(bool & val, const std::string & raw_str)
{
  std::string str_val;
  auto success = parse_<std::string>(str_val, raw_str);
  if (!success)
    return false;

  if (str_val == "true")
  {
    val = true;
    return true;
  }

  if (str_val == "false")
  {
    val = false;
    return true;
  }

  return false;
}

template <>
bool
parse_vector_(std::vector<bool> & vals, const std::string & raw_str)
{
  auto tokens = split(raw_str, " \t\n\v\f\r");
  vals.resize(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); i++)
  {
    bool val = false;
    auto success = parse_<bool>(val, tokens[i]);
    if (!success)
      return false;
    vals[i] = val;
  }
  return true;
}

template <>
bool
parse_(VariableName & val, const std::string & raw_str)
{
  auto tokens = split(raw_str, "/ \t\n\v\f\r");
  val = VariableName(tokens);
  return true;
}

template <>
bool
parse_(TensorShape & val, const std::string & raw_str)
{
  if (!start_with(raw_str, "(") || !end_with(raw_str, ")"))
    return false;

  auto inner = trim(raw_str, "() \t\n\v\f\r");
  auto tokens = split(inner, ", \t\n\v\f\r");

  val.resize(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); i++)
  {
    auto success = parse_<Size>(val[i], tokens[i]);
    if (!success)
      return false;
  }
  return true;
}

template <>
Device
parse(const std::string & raw_str)
{
  return Device(parse<std::string>(raw_str));
}

template <>
bool
parse_(Device & val, const std::string & raw_str)
{
  const auto str_val = parse<std::string>(raw_str);
  try
  {
    val = Device(str_val);
    return true;
  }
  catch (const std::exception & e)
  {
    std::cerr << "Failed to parse '" << raw_str << "' as a device. Error message:\n"
              << e.what() << std::endl;
    return false;
  }
}
} // namespace utils
} // namespace neml2
