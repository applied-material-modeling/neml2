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

#pragma once

#include "neml2/user_tensors/UserTensorBase.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
/**
 * @brief Create a Gaussian tensor evaluated at provided points.
 *
 * @tparam T The concrete tensor derived from TensorBase
 */
template <typename T>
class GaussianTensorTmpl : public UserTensorBase<T>
{
public:
  static OptionSet expected_options();

  GaussianTensorTmpl(const OptionSet & options);

protected:
  T make() const override;

private:
  /// The coordinates to evaluate the Gaussian at
  const TensorName<T> _points;

  /// The Gaussian width
  const TensorName<Scalar> _width;

  /// The Gaussian height
  const TensorName<Scalar> _height;

  /// The Gaussian center
  const TensorName<Scalar> _center;

  // Note: No linspace parameters are required; points are provided directly.
};
} // namespace neml2
