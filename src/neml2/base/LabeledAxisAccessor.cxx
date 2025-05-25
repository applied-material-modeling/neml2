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

#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/misc/assertions.h"
#include "neml2/misc/string_utils.h"

namespace neml2
{
std::vector<std::string>
reserved_subaxis_names()
{
  return {STATE, OLD_STATE, FORCES, OLD_FORCES, RESIDUAL, PARAMETERS};
}

LabeledAxisAccessor::operator std::vector<std::string>() const
{
  std::vector<std::string> v(_item_names.begin(), _item_names.end());
  return v;
}

std::string
LabeledAxisAccessor::str() const
{
  return utils::stringify(*this);
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
  std::vector<std::string> new_names(_item_names.begin() + n, _item_names.end());
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::slice(int64_t n1, int64_t n2) const
{
  n1 = n1 < 0 ? int64_t(size()) + n1 : n1;
  n2 = n2 < 0 ? int64_t(size()) + n2 : n2;
  neml_assert(size() >= std::size_t(n1), "cannot apply slice");
  neml_assert(size() >= std::size_t(n2), "cannot apply slice");
  std::vector<std::string> new_names(_item_names.begin() + n1, _item_names.begin() + n2);
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::remount(const LabeledAxisAccessor & axis, int64_t n) const
{
  return slice(n).prepend(axis);
}

bool
LabeledAxisAccessor::start_with(const LabeledAxisAccessor & axis) const
{
  return slice(0, int64_t(axis.size())) == axis;
}

bool
LabeledAxisAccessor::is_state() const
{
  return start_with(STATE);
}

bool
LabeledAxisAccessor::is_old_state() const
{
  return start_with(OLD_STATE);
}

bool
LabeledAxisAccessor::is_force() const
{
  return start_with(FORCES);
}

bool
LabeledAxisAccessor::is_old_force() const
{
  return start_with(OLD_FORCES);
}

bool
LabeledAxisAccessor::is_residual() const
{
  return start_with(RESIDUAL);
}

bool
LabeledAxisAccessor::is_parameter() const
{
  return start_with(PARAMETERS);
}

LabeledAxisAccessor
LabeledAxisAccessor::current() const
{
  neml_assert(_item_names.size() >= 1, "variable name length must be at least 1");
  if (start_with(OLD_STATE))
    return remount(STATE);
  if (start_with(OLD_FORCES))
    return remount(FORCES);
  throw NEMLException("Unable to find current counterpart of variable named '" +
                      utils::stringify(*this) + "'");
}

LabeledAxisAccessor
LabeledAxisAccessor::old() const
{
  neml_assert(_item_names.size() >= 1, "variable name length must be at least 1");
  if (start_with(STATE))
    return remount(OLD_STATE);
  if (start_with(FORCES))
    return remount(OLD_FORCES);
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
