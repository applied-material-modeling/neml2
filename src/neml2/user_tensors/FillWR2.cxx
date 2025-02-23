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

#include "neml2/user_tensors/FillWR2.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(FillWR2);

OptionSet
FillWR2::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a Rot from a vector of Scalars.";

  options.set<std::vector<TensorName<Scalar>>>("values");
  options.set("values").doc() = "Scalars used to fill the WR2";

  return options;
}

FillWR2::FillWR2(const OptionSet & options)
  : WR2(fill(options.get<std::vector<TensorName<Scalar>>>("values"))),
    UserTensorBase(options)
{
}

WR2
FillWR2::fill(const std::vector<TensorName<Scalar>> & values) const
{
  neml_assert(values.size() == 3,
              "Number of values must be 3, but ",
              values.size(),
              " values are provided.");

  return WR2::fill(values[0].resolve(), values[1].resolve(), values[2].resolve());
}
} // namespace neml2
