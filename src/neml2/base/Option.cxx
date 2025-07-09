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

#include "neml2/base/Option.h"
#include "neml2/misc/string_utils.h"
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/base/EnumSelection.h"
#include "neml2/base/MultiEnumSelection.h"
#include "neml2/base/TensorName.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/base/Factory.h"
#include "neml2/base/Settings.h"

namespace neml2
{
template <typename T>
Option<T>::Option(const std::string & name)
  : _value()
{
  _metadata.name = name;
  _metadata.type = utils::demangle(typeid(T).name());
}

template <>
Option<Device>::Option(const std::string & name)
  : _value(kCPU)
{
  _metadata.name = name;
  _metadata.type = utils::demangle(typeid(Device).name());
}

template <typename T>
bool
Option<T>::operator==(const OptionBase & other) const
{
  const auto other_ptr = dynamic_cast<const Option<T> *>(&other);
  if (!other_ptr)
    return false;

  return (_metadata == other_ptr->_metadata) && (other_ptr->get() == this->get());
}

template <typename T>
bool
Option<T>::operator!=(const OptionBase & other) const
{
  return !(*this == other);
}

// LCOV_EXCL_START
template <typename T>
void
Option<T>::print(std::ostream & os) const
{
  details::_print_helper(os, static_cast<const T *>(&_value));
}
// LCOV_EXCL_STOP

template <typename T>
std::unique_ptr<OptionBase>
Option<T>::clone() const
{
  auto copy = std::make_unique<Option<T>>(this->name());
  copy->_value = this->_value;
  copy->_metadata = this->_metadata;
  return copy;
}

template class Option<bool>;
template class Option<std::vector<bool>>;
template class Option<std::vector<std::vector<bool>>>;

template class Option<int>;
template class Option<std::vector<int>>;
template class Option<std::vector<std::vector<int>>>;

template class Option<unsigned int>;
template class Option<std::vector<unsigned int>>;
template class Option<std::vector<std::vector<unsigned int>>>;

template class Option<std::size_t>;
template class Option<std::vector<std::size_t>>;
template class Option<std::vector<std::vector<std::size_t>>>;

template class Option<Size>;
template class Option<std::vector<Size>>;
template class Option<std::vector<std::vector<Size>>>;

template class Option<double>;
template class Option<std::vector<double>>;
template class Option<std::vector<std::vector<double>>>;

template class Option<std::string>;
template class Option<std::vector<std::string>>;
template class Option<std::vector<std::vector<std::string>>>;

template class Option<VariableName>;
template class Option<std::vector<VariableName>>;
template class Option<std::vector<std::vector<VariableName>>>;

template class Option<TensorShape>;
template class Option<std::vector<TensorShape>>;
template class Option<std::vector<std::vector<TensorShape>>>;

template class Option<TensorName<ATensor>>;
template class Option<std::vector<TensorName<ATensor>>>;
template class Option<std::vector<std::vector<TensorName<ATensor>>>>;

#define INSTANTIATE_TENSORNAME(T)                                                                  \
  template class Option<TensorName<T>>;                                                            \
  template class Option<std::vector<TensorName<T>>>;                                               \
  template class Option<std::vector<std::vector<TensorName<T>>>>
FOR_ALL_TENSORBASE(INSTANTIATE_TENSORNAME);

template class Option<Device>;
template class Option<std::vector<Device>>;
template class Option<std::vector<std::vector<Device>>>;

// Special instantiations
template class Option<EnumSelection>;
template class Option<MultiEnumSelection>;
template class Option<Factory *>;
template class Option<std::shared_ptr<Settings>>;
template class Option<NEML2Object *>;

namespace details
{
template <>
void
_print_helper(std::ostream & os, const char * option)
{
  os << static_cast<int>(*option);
}

template <>
void
_print_helper(std::ostream & os, const unsigned char * option)
{
  os << static_cast<int>(*option);
}

template <>
void
_print_helper(std::ostream & os, const bool * option)
{
  os << (*option ? "true" : "false");
}

template <>
void
_print_helper(std::ostream & os, const std::vector<bool> * option)
{
  for (std::size_t i = 0; i < option->size(); i++)
  {
    if (i > 0)
      os << " ";
    os << ((*option)[i] ? "true" : "false");
  }
}

template <>
void
_print_helper(std::ostream & os, const TensorShape * option)
{
  os << "(";
  for (std::size_t i = 0; i < option->size(); i++)
  {
    if (i > 0)
      os << ",";
    os << (*option)[i];
  }
  os << ")";
}
} // namespace details
} // namespace neml2
