// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/base/ParameterCollection.h"

namespace neml2
{
void
ParameterCollection::merge(ParameterCollection && other)
{
  for (auto & [section, objects] : other._data)
    if (_data.count(section))
      _data[section].merge(other[section]);
    else
      _data[section] = other[section];
}

std::map<std::string, ParameterSet> &
ParameterCollection::operator[](const std::string & section)
{
  return _data[section];
}

const std::map<std::string, ParameterSet> &
ParameterCollection::operator[](const std::string & section) const
{
  return _data.at(section);
}

std::ostream &
operator<<(std::ostream & os, const ParameterCollection & p)
{
  size_t width = 79;
  auto toprule = std::string(width, '=');
  auto midrule = std::string(width, '-');
  auto bottomrule = toprule;

  for (auto && [section, obj_params] : p.data())
  {
    os << toprule << std::endl;
    os << section << std::endl;
    os << toprule << std::endl;
    for (auto && [obj, params] : obj_params)
      os << obj << std::endl << midrule << std::endl << params << std::endl << midrule << std::endl;
  }
  return os;
}
} // namespace neml2