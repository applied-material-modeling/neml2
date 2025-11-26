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

#include "neml2/models/LinearInterpolation.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/indexing.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
template <typename T>
OptionSet
LinearInterpolation<T>::expected_options()
{
  OptionSet options = Interpolation<T>::expected_options();
  options.doc() += " This object performs a _linear interpolation_.";

  options.set<bool>("define_second_derivatives") = true;

  options.set<TensorName<Scalar>>("abscissa");
  options.set("abscissa").doc() = "Scalar defining the abscissa values of the interpolant";

  options.set<Size>("dim") = -1;
  options.set("dim").doc() = "Intermediate dimension along which to interpolate";

  options.set_input("argument");
  options.set("argument").doc() = "Argument used to query the interpolant";

  return options;
}

template <typename T>
LinearInterpolation<T>::LinearInterpolation(const OptionSet & options)
  : Interpolation<T>(options),
    _X(this->template declare_parameter<Scalar>("X", "abscissa")),
    _x(this->template declare_input_variable<Scalar>("argument")),
    _dim(options.get<Size>("dim"))
{
}

template <typename T>
void
LinearInterpolation<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  using namespace indexing;

  neml_assert_dbg(this->_X.intmd_size(_dim) == this->_Y.intmd_size(_dim),
                  "Abscissa and ordinate intermediate dimension ",
                  _dim,
                  " must have the same size (",
                  this->_X.intmd_size(_dim),
                  " vs ",
                  this->_Y.intmd_size(_dim),
                  ").");

  // First move the interpolating dimension to the end
  const auto X = this->_X.intmd_movedim(_dim, -1);
  const auto Y = this->_Y.intmd_movedim(_dim, -1);

  // Unsqueeze one intermediate dimension in x to match X and Y
  const auto x = this->_x().intmd_unsqueeze(-1);

  // Get the mask for the interpolating interval
  // Also transform x onto the parametric space (0, 1]
  const auto [m, xi, dxi_dx] = parametric_coordinates(X, x);

  // Get the ordinate interval to be interpolated
  auto Y1 = Y.intmd_slice(-1, Slice(None, -1));
  auto Y2 = Y.intmd_slice(-1, Slice(1));
  Y1 = apply_mask(Y1, m, 1);
  Y2 = apply_mask(Y2, m, 1);
  auto dY = Y2 - Y1;

  if (out)
    this->_p = Y1 + xi * dY;

  if (dout_din)
    if (this->_x.is_dependent())
      this->_p.d(this->_x) = dxi_dx * dY;

  if (d2out_din2)
  {
    // zero
  }
}

#define REGISTER(T)                                                                                \
  using T##LinearInterpolation = LinearInterpolation<T>;                                           \
  register_NEML2_object(T##LinearInterpolation);                                                   \
  template class LinearInterpolation<T>
REGISTER(Scalar);
REGISTER(Vec);
REGISTER(SR2);
} // namespace neml2
