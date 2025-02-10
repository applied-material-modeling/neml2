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

#include "neml2/models/NonlinearParameter.h"

namespace neml2
{
class Scalar;

/**
 * @brief The base class for interpolated nonlinear parameter
 *
 * This model requires two parameters, namely the "abscissa" and the "ordinate". The ordinate is
 * interpolated using an input (specified by the "argument" option) along the axis of abscissa.
 *
 * The interpolant's batch shape is defined as the broadcasted batch shapes of the abscissa and the
 * ordinate, after excluding the dimensions on which the interpolation happens.
 *
 * The general expectations for the batch shapes are:
 * 1. The abscissa and the ordinate should be batch-broadcastable.
 * 2. The abscissa should always be a Scalar. The ordinate can be of any primitive tensor type.
 * 3. The input (specified by option "argument") must be a Scalar.
 * 4. The input and the interpolant should be batch-broadcastable.
 * 5. Broadcasting the input with the interpolant should not alter its batch shape.
 */
template <typename T>
class Interpolation : public NonlinearParameter<T>
{
public:
  static OptionSet expected_options();

  Interpolation(const OptionSet & options);

protected:
  /// The abscissa values of the interpolant
  const Scalar & _X;

  /// The ordinate values of the interpolant
  const T & _Y;

  /// Argument of interpolation
  const Variable<Scalar> & _x;
};
} // namespace neml2
