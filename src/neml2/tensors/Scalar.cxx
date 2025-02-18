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

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
Scalar::Scalar(Real init, const TensorOptions & options)
  : Scalar(at::scalar_tensor(init, options))
{
  neml_assert_dbg(
      !options.requires_grad(),
      "When creating a Scalar from a Real, requires_grad must be false. If you are "
      "trying to create a Scalar as a leaf variable, use Scalar::create or Scalar::full.");
}

Scalar
Scalar::identity_map(const TensorOptions & options)
{
  return Scalar::ones(options);
}

neml2::Tensor
Scalar::base_unsqueeze_to(Size n) const
{
  indexing::TensorIndices net{indexing::Ellipsis};
  net.insert(net.end(), n, indexing::None);
  return neml2::Tensor(index(net), batch_sizes());
}
} // namespace neml2
