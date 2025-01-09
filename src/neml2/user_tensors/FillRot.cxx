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

#include "neml2/user_tensors/FillRot.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{
register_NEML2_object(FillRot);

OptionSet
FillRot::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a Rot from a vector of Scalars.";

  options.set<std::vector<TensorName>>("values");
  options.set("values").doc() = "Scalars used to fill the Rot";

  options.set<std::string>("method") = "modified";
  options.set("method").doc() = "Fill method, options are 'modified' and 'standard'.";

  return options;
}

FillRot::FillRot(const OptionSet & options)
  : Rot(fill(options.get<std::vector<TensorName>>("values"), options.get<std::string>("method"))),
    UserTensorBase(options)
{
}

Rot
FillRot::fill(const std::vector<TensorName> & values, const std::string & method) const
{
  if (method == "modified")
  {
    neml_assert(values.size() == 3,
                "Number of values must be 3, but ",
                values.size(),
                " values are provided.");
    return Rot::fill(Scalar(values[0]), Scalar(values[1]), Scalar(values[2]));
  }

  if (method == "standard")
  {
    neml_assert(values.size() == 3,
                "Number of values must be 3, but ",
                values.size(),
                " values are provided.");
    auto ns = Scalar(values[0]) * Scalar(values[0]) + Scalar(values[1]) * Scalar(values[1]) +
              Scalar(values[2]) * Scalar(values[2]);
    auto f = neml2::sqrt(ns + 1.0) + 1.0;
    return Rot::fill(Scalar(values[0]) / f, Scalar(values[1]) / f, Scalar(values[2]) / f);
  }

  throw NEMLException("Unknown Rot fill type " + method);
}
} // namespace neml2
