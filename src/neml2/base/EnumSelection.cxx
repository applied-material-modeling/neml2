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
  es.select(ss.str());
  return ss;
}

EnumSelection::EnumSelection(const EnumSelection & other)
  : EnumSelectionBase(other),
    _selection(other._selection),
    _value(other._value)
{
}

EnumSelection::EnumSelection(const std::vector<std::string> & candidates,
                             const std::string & selection)
  : EnumSelectionBase(candidates)
{
  select(selection);
}

EnumSelection::EnumSelection(const std::vector<std::string> & candidates,
                             const std::vector<int> & values,
                             const std::string & selection)
  : EnumSelectionBase(candidates, values)
{
  select(selection);
}

EnumSelection &
EnumSelection::operator=(const EnumSelection & other)
{
  _candidate_map = other._candidate_map;
  _selection = other._selection;
  _value = other._value;
  return *this;
}

bool
EnumSelection::operator==(const EnumSelection & other) const
{
  return _candidate_map == other._candidate_map && _selection == other._selection &&
         _value == other._value;
}

bool
EnumSelection::operator!=(const EnumSelection & other) const
{
  return !(*this == other);
}

void
EnumSelection::select(const std::string & selection)
{
  neml_assert(_candidate_map.count(selection),
              "Invalid selection for EnumSelection. Candidates are ",
              candidates_str());
  _selection = selection;
  _value = _candidate_map[selection];
}
} // namesace neml2
