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

#include <set>

#include "neml2/base/EnumSelectionBase.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
EnumSelectionBase::EnumSelectionBase(const std::vector<std::string> & choices)
{
  std::set<std::string> choices_set(choices.begin(), choices.end());
  neml_assert(choices_set.size() == choices.size(),
              "Choices of (Multi)EnumSelection must be unique.");

  int count = 0;
  for (const auto & candidate : choices)
    _choice_map.emplace(candidate, count++);
}

EnumSelectionBase::EnumSelectionBase(const std::vector<std::string> & choices,
                                     const std::vector<int> & values)
{
  neml_assert(choices.size() == values.size(),
              "In (Multi)EnumSelection, number of choices must match the number of values.");

  std::set<std::string> choices_set(choices.begin(), choices.end());
  neml_assert(choices_set.size() == choices.size(),
              "Choices of (Multi)EnumSelection must be unique.");

  std::set<int> values_set(values.begin(), values.end());
  neml_assert(values_set.size() == values.size(), "Values of (Multi)EnumSelection must be unique.");

  for (size_t i = 0; i < choices.size(); i++)
    _choice_map.emplace(choices[i], values[i]);
}

std::string
EnumSelectionBase::join(const std::string & separator) const
{
  std::stringstream ss;
  std::size_t count = 0;
  for (const auto & [e, v] : _choice_map)
  {
    ss << e;
    if (count < _choice_map.size() - 1)
      ss << separator;
    count++;
  }
  return ss.str();
}
} // namespace neml2
