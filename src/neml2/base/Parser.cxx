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
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/misc/errors.h"
#include "neml2/misc/types.h"

#include "neml2/misc/string_utils.h"
#include "neml2/misc/errors.h"

namespace neml2
{
const std::vector<std::string> Parser::sections = {
    "Tensors", "EquationSystems", "Solvers", "Data", "Models", "Drivers", "Schedulers"};

namespace utils
{

template <>
TensorShape
parse<TensorShape>(const std::string & raw_str)
{
  if (!start_with(raw_str, "(") || !end_with(raw_str, ")"))
    throw ParserException("Invalid tensor shape: " + raw_str +
                          ". Tensor shapes must begin with '(' and end with ')'.");

  auto inner = trim(raw_str, "()");
  auto tokens = split(inner, ",");

  TensorShape val(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); i++)
  {
    std::size_t pos = 0;
    try
    {
      val[i] = std::stoll(tokens[i], &pos);
      if (pos != tokens[i].size())
        throw ParserException("Invalid integer value '" + tokens[i] +
                              "' in tensor shape: " + raw_str);
    }
    catch (...)
    {
      throw ParserException("Invalid integer value '" + tokens[i] +
                            "' in tensor shape: " + raw_str);
    }
  }
  return val;
}

template <>
VariableName
parse<VariableName>(const std::string & raw_str)
{
  auto tokens = split(raw_str, "/");
  try
  {
    return VariableName(tokens);
  }
  catch (const NEMLException & err)
  {
    throw ParserException(err.what());
  }
}

template <>
Device
parse<Device>(const std::string & raw_str)
{
  try
  {
    return Device(raw_str);
  }
  catch (...)
  {
    throw ParserException("Invalid device spec: " + raw_str);
  }
}

} // namespace utils
} // namespace neml2
