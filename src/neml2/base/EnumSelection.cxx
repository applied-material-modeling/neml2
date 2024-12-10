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

#include "neml2/base/EnumSelection.h"
#include "neml2/misc/parser_utils.h"

namespace neml2
{
std::ostream &
operator<<(std::ostream & os, const EnumSelection & es)
{
  os << std::string(es);
  return os;
}

std::stringstream &
operator>>(std::stringstream & ss, EnumSelection & es)
{
  ss >> es._selection;

  if (!es._values.count(es._selection))
    throw ParserException("Invalid selection '" + es._selection +
                          "', candidates are: " + es.candidates_str());

  // Also update the enum integral value
  es._value = es._values[es._selection];

  return ss;
}

EnumSelection::EnumSelection(const EnumSelection & other)
  : _values(other._values),
    _selection(other._selection),
    _value(other._value)
{
}

EnumSelection::EnumSelection(const std::vector<std::string> & candidates,
                             const std::string & selection)
  : _selection(selection)
{
  std::set<std::string> candidates_set(candidates.begin(), candidates.end());
  neml_assert(candidates_set.size() == candidates.size(),
              "Candidates of EnumSelection must be unique.");

  int count = 0;
  for (const auto & candidate : candidates)
    _values.emplace(candidate, count++);

  neml_assert(_values.count(_selection),
              "Invalid default selection for EnumSelection. Candidates are ",
              candidates_str());

  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  _value = _values[_selection];
}

EnumSelection::EnumSelection(const std::vector<std::string> & candidates,
                             const std::vector<int> & values,
                             const std::string & selection)
  : _selection(selection)
{
  neml_assert(candidates.size() == values.size(),
              "In EnumSelection, number of candidates must match the number of values.");

  std::set<std::string> candidates_set(candidates.begin(), candidates.end());
  neml_assert(candidates_set.size() == candidates.size(),
              "Candidates of EnumSelection must be unique.");

  std::set<int> values_set(values.begin(), values.end());
  neml_assert(values_set.size() == values.size(), "Values of EnumSelection must be unique.");

  for (size_t i = 0; i < candidates.size(); i++)
    _values.emplace(candidates[i], values[i]);

  neml_assert(_values.count(_selection),
              "Invalid default selection for EnumSelection. Candidates are ",
              candidates_str());

  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  _value = _values[_selection];
}

EnumSelection &
EnumSelection::operator=(const EnumSelection & other)
{
  _values = other._values;
  _selection = other._selection;
  _value = other._value;
  return *this;
}

bool
EnumSelection::operator==(const EnumSelection & other) const
{
  return _values == other._values && _selection == other._selection && _value == other._value;
}

bool
EnumSelection::operator!=(const EnumSelection & other) const
{
  return !(*this == other);
}

bool
EnumSelection::operator==(const std::string & other) const
{
  return _selection == other;
}

bool
EnumSelection::operator!=(const std::string & other) const
{
  return _selection != other;
}

std::string
EnumSelection::candidates_str() const
{
  std::stringstream ss;
  for (const auto & [e, v] : _values)
    ss << e << " ";
  return ss.str();
}
} // namesace neml2
