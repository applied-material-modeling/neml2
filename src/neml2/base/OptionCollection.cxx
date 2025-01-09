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

#include "neml2/base/OptionCollection.h"
#include "neml2/base/Settings.h"

namespace neml2
{
OptionCollection::OptionCollection()
  : _settings(Settings::expected_options())
{
}

std::map<std::string, OptionSet> &
OptionCollection::operator[](const std::string & section)
{
  return _data[section];
}

const std::map<std::string, OptionSet> &
OptionCollection::operator[](const std::string & section) const
{
  return _data.at(section);
}

// LCOV_EXCL_START
std::ostream &
operator<<(std::ostream & os, const OptionCollection & p)
{
  size_t width = 79;
  auto toprule = std::string(width, '=');
  auto midrule = std::string(width, '-');

  for (auto && [section, obj_options] : p.data())
  {
    os << toprule << std::endl;
    os << section << std::endl;
    os << toprule << std::endl;
    for (auto && [obj, options] : obj_options)
      os << obj << std::endl
         << midrule << std::endl
         << options << std::endl
         << midrule << std::endl;
  }
  return os;
}
// LCOV_EXCL_STOP
} // namespace neml2
