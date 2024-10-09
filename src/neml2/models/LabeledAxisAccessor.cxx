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

#include "neml2/models/LabeledAxisAccessor.h"
#include "neml2/misc/error.h"
#include "neml2/misc/parser_utils.h"

namespace neml2
{
LabeledAxisAccessor::operator std::vector<std::string>() const
{
  std::vector<std::string> v(_item_names.begin(), _item_names.end());
  return v;
}

bool
LabeledAxisAccessor::empty() const
{
  return _item_names.empty();
}

size_t
LabeledAxisAccessor::size() const
{
  return _item_names.size();
}

const std::string &
LabeledAxisAccessor::operator[](size_t i) const
{
  return _item_names[i];
}

LabeledAxisAccessor
LabeledAxisAccessor::with_suffix(const std::string & suffix) const
{
  auto new_names = _item_names;
  new_names.back() += suffix;
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::append(const LabeledAxisAccessor & axis) const
{
  return axis.prepend(*this);
}

LabeledAxisAccessor
LabeledAxisAccessor::prepend(const LabeledAxisAccessor & axis) const
{
  auto new_names = axis._item_names;
  new_names.insert(new_names.end(), _item_names.begin(), _item_names.end());
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::slice(int64_t n) const
{
  n = n < 0 ? int64_t(size()) + n : n;
  neml_assert(size() >= std::size_t(n), "cannot apply slice");
  c10::SmallVector<std::string> new_names(_item_names.begin() + n, _item_names.end());
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::slice(int64_t n1, int64_t n2) const
{
  n1 = n1 < 0 ? int64_t(size()) + n1 : n1;
  n2 = n2 < 0 ? int64_t(size()) + n2 : n2;
  neml_assert(size() >= std::size_t(n1), "cannot apply slice");
  neml_assert(size() >= std::size_t(n2), "cannot apply slice");
  c10::SmallVector<std::string> new_names(_item_names.begin() + n1, _item_names.begin() + n2);
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::remount(const LabeledAxisAccessor & axis, Size n) const
{
  return slice(n).prepend(axis);
}

bool
LabeledAxisAccessor::start_with(const LabeledAxisAccessor & axis) const
{
  return slice(0, int64_t(axis.size())) == axis;
}

LabeledAxisAccessor
LabeledAxisAccessor::old() const
{
  // NOLINTNEXTLINE(readability-container-size-empty)
  neml_assert(_item_names.size() >= 1, "variable name length must be at least 1");
  if (start_with("state"))
    return remount("old_state");
  if (start_with("forces"))
    return remount("old_forces");
  throw NEMLException("Unable to find old counterpart of variable named '" +
                      utils::stringify(*this) + "'");
}

void
LabeledAxisAccessor::validate_item_name(const std::string & name) const
{
  neml_assert(!name.empty(), "Empty item variable name");
  const auto x = name.find_first_of(" .,;/\t\n\v\f\r");
  neml_assert(x == std::string::npos,
              "Invalid item name: ",
              name,
              ". The item names cannot contain whitespace, '.', ',', ';', or '/'.");
}

bool
operator!=(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b)
{
  return a.vec() != b.vec();
}

bool
operator==(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b)
{
  return a.vec() == b.vec();
}

bool
operator<(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b)
{
  return a.vec() < b.vec();
}

std::ostream &
operator<<(std::ostream & os, const LabeledAxisAccessor & accessor)
{
  for (size_t i = 0; i < accessor.vec().size(); i++)
  {
    if (i != 0)
      os << "/";
    os << accessor.vec()[i];
  }
  return os;
}
} // namespace neml2
