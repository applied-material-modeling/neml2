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

#include "neml2/base/Parser.h"
#include "neml2/base/OptionCollection.h"
#include "neml2/models/Variable.h"

namespace neml2
{
const std::vector<std::string> Parser::sections = {
    "Tensors", "Solvers", "Data", "Models", "Drivers", "Schedulers"};

namespace utils
{
template <>
void
parse_(bool & val, const std::string & raw_str)
{
  const auto str_val = parse<std::string>(raw_str);
  if (str_val == "true")
    val = true;
  else if (str_val == "false")
    val = false;
  else
    throw ParserException("Failed to parse boolean value. Only 'true' and 'false' are recognized.");
}

template <>
void
parse_vector_(std::vector<bool> & vals, const std::string & raw_str)
{
  auto tokens = split(raw_str, " \t\n\v\f\r");
  vals.resize(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++)
    vals[i] = parse<bool>(tokens[i]);
}

template <>
void
parse_(VariableName & val, const std::string & raw_str)
{
  auto tokens = split(raw_str, "/ \t\n\v\f\r");
  val = VariableName(tokens);
}

template <>
void
parse_(TensorShape & val, const std::string & raw_str)
{
  if (!start_with(raw_str, "(") || !end_with(raw_str, ")"))
    throw ParserException("Trying to parse " + raw_str +
                          " as a shape, but a shape must start with '(' and end with ')'");

  auto inner = trim(raw_str, "() \t\n\v\f\r");
  auto tokens = split(inner, ", \t\n\v\f\r");

  val.clear();
  for (auto & token : tokens)
    val.push_back(parse<Size>(token));
}

template <>
void
parse_(Device & val, const std::string & raw_str)
{
  const auto str_val = parse<std::string>(raw_str);
  val = Device(str_val);
}
} // namespace utils
} // namespace neml2
