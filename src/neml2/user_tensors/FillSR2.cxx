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

#include "neml2/user_tensors/FillSR2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(FillSR2);

OptionSet
FillSR2::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a R2 with a vector of Scalars. The vector length must be 1, 3, or 6. "
                  "See the full documentation on neml2::SR2 on the dispatch of fill method. When "
                  "vector length is 1, the Scalar value is used to fill the diagonals; when vector "
                  "length is 3, the Scalar values are used to fill the respective diagonal "
                  "entries; when vector length is 6, the Scalar values are used to fill the tensor "
                  "following the Voigt notation.";

  options.set<std::vector<TensorName<Scalar>>>("values");
  options.set("values").doc() = "Scalars used to fill the R2";

  return options;
}

FillSR2::FillSR2(const OptionSet & options)
  : UserTensorBase(options),
    SR2(fill(options.get<std::vector<TensorName<Scalar>>>("values")))
{
}

SR2
FillSR2::fill(const std::vector<TensorName<Scalar>> & values) const
{
  auto * f = factory();
  neml_assert(f, "Internal error: factory != nullptr");

  if (values.size() == 1)
    return SR2::fill(values[0].resolve(f));
  if (values.size() == 3)
    return SR2::fill(values[0].resolve(f), values[1].resolve(f), values[2].resolve(f));
  if (values.size() == 6)
    return SR2::fill(values[0].resolve(f),
                     values[1].resolve(f),
                     values[2].resolve(f),
                     values[3].resolve(f),
                     values[4].resolve(f),
                     values[5].resolve(f));

  throw NEMLException("Number of values must be 1, 3, or 6, but " + std::to_string(values.size()) +
                      " values are provided.");
}
} // namespace neml2
