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

#include "neml2/base/MultiEnumSelection.h"
#include "neml2/base/Parser.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
std::ostream &
operator<<(std::ostream & os, const MultiEnumSelection & es)
{
  auto selections = std::vector<std::string>(es);
  for (size_t i = 0; i < selections.size(); i++)
  {
    os << selections[i];
    if (i < selections.size() - 1)
      os << " ";
  }
  return os;
}

std::stringstream &
operator>>(std::stringstream & ss, MultiEnumSelection & es)
{
  es.select(utils::parse_vector<std::string>(ss.str()));
  return ss;
}

MultiEnumSelection::MultiEnumSelection(const std::vector<std::string> & candidates,
                                       const std::vector<std::string> & selections)
  : EnumSelectionBase(candidates)
{
  select(selections);
}

MultiEnumSelection::MultiEnumSelection(const std::vector<std::string> & candidates,
                                       const std::vector<int> & values,
                                       const std::vector<std::string> & selections)
  : EnumSelectionBase(candidates, values)
{
  select(selections);
}

bool
MultiEnumSelection::operator==(const MultiEnumSelection & other) const
{
  return _candidate_map == other._candidate_map && _selections == other._selections &&
         _values == other._values;
}

bool
MultiEnumSelection::operator!=(const MultiEnumSelection & other) const
{
  return !(*this == other);
}

void
MultiEnumSelection::select(const std::vector<std::string> & selections)
{
  _selections.clear();
  _values.clear();
  for (const auto & selection : selections)
  {
    neml_assert(_candidate_map.count(selection),
                "Invalid selection for MultiEnumSelection. Candidates are ",
                candidates_str());
    _selections.push_back(selection);
    _values.push_back(_candidate_map[selection]);
  }
}
} // namesace neml2
