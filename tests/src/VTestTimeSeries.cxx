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

#include "VTestTimeSeries.h"
#include "VTestParser.h"

#include "utils.h"

namespace neml2
{
template <typename T>
OptionSet
VTestTimeSeries<T>::expected_options()
{
  OptionSet options = NEML2Object::expected_options();
  options.set<std::string>("vtest");
  options.set<std::string>("variable");
  return options;
}

template <typename T>
VTestTimeSeries<T>::VTestTimeSeries(const OptionSet & options)
  : NEML2Object(options),
    T(init(options))
{
}

template <typename T>
T
VTestTimeSeries<T>::init(const OptionSet & /*options*/) const
{
  throw NEMLException(name() + " has not been implemented");
  return T();
}

template <>
Scalar
VTestTimeSeries<Scalar>::init(const OptionSet & options) const
{
  VTestParser table(options.get<std::string>("vtest"));
  auto var = options.get<std::string>("variable");
  return Scalar(table[var]);
}

template <>
SR2
VTestTimeSeries<SR2>::init(const OptionSet & options) const
{
  VTestParser table(options.get<std::string>("vtest"));
  auto var = options.get<std::string>("variable");
  auto val_xx = table[var + "_xx"];
  auto val_yy = table[var + "_yy"];
  auto val_zz = table[var + "_zz"];
  auto val_yz = table[var + "_yz"];
  auto val_xz = table[var + "_xz"];
  auto val_xy = table[var + "_xy"];
  return base_stack({val_xx, val_yy, val_zz, val_yz, val_xz, val_xy}, -1);
}

template <>
WR2
VTestTimeSeries<WR2>::init(const OptionSet & options) const
{
  VTestParser table(options.get<std::string>("vtest"));
  auto var = options.get<std::string>("variable");
  auto val_zy = table[var + "_zy"];
  auto val_xz = table[var + "_xz"];
  auto val_yx = table[var + "_yx"];
  return base_stack({val_zy, val_xz, val_yx}, -1);
}

#define REGISTER_VTESTTIMESERIES(T)                                                                \
  using T##VTestTimeSeries = VTestTimeSeries<T>;                                                   \
  register_NEML2_object(T##VTestTimeSeries)
FOR_ALL_PRIMITIVETENSOR(REGISTER_VTESTTIMESERIES);
} // namespace neml2
