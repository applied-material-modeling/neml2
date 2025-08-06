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

#include "neml2/models/BilinearInterpolation.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/functions/diff.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{
template <typename T>
OptionSet
BilinearInterpolation<T>::expected_options()
{
  OptionSet options = Interpolation<T>::expected_options();
  options.doc() += " This object performs a _bilinear interpolation_.";

  options.set<bool>("define_second_derivatives") = true;

  options.set<TensorName<Scalar>>("abscissa1");
  options.set("abscissa1").doc() =
      "Scalar defining the abscissa values of the first interpolation axis";

  options.set<TensorName<Scalar>>("abscissa2");
  options.set("abscissa2").doc() =
      "Scalar defining the abscissa values of the second interpolation axis";

  options.set_input("argument1");
  options.set("argument1").doc() =
      "First argument used to query the interpolant along the first axis";

  options.set_input("argument2");
  options.set("argument2").doc() =
      "Second argument used to query the interpolant along the second axis";

  return options;
}

template <typename T>
BilinearInterpolation<T>::BilinearInterpolation(const OptionSet & options)
  : Interpolation<T>(options),
    _X1(this->template declare_parameter<Scalar>("X1", "abscissa1")),
    _X2(this->template declare_parameter<Scalar>("X2", "abscissa2")),
    _x1(this->template declare_input_variable<Scalar>("argument1")),
    _x2(this->template declare_input_variable<Scalar>("argument2"))
{
}

static std::tuple<Scalar, Scalar, Scalar>
parametric_coordinates(const Scalar & X, const Scalar & x)
{
  using namespace indexing;
  const auto m =
      at::logical_and(at::gt(x.batch_unsqueeze(-1), X.index({Ellipsis, Slice(None, -1)})),
                      at::le(x.batch_unsqueeze(-1), X.index({Ellipsis, Slice(1)})));
  const auto X_start = X.index({Ellipsis, Slice(None, -1)}).expand_as(m).index({m});
  const auto X_end = X.index({Ellipsis, Slice(1)}).expand_as(m).index({m});
  const auto xi = (x - X_start) / (X_end - X_start);
  const auto dxi = 1.0 / (X_end - X_start);
  return {Scalar(m), Scalar(xi), Scalar(dxi)};
}

template <typename T>
static T
apply_mask(const T & y, const Scalar & m)
{
  const auto B = utils::broadcast_batch_sizes({m, y});
  const auto D = B.slice(0, -2); // excluding the interpolation grid
  return T(y.batch_expand(B).index({m.batch_expand(B)})).batch_reshape(D);
}

template <typename T>
void
BilinearInterpolation<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  using namespace indexing;

  // Get masks for the interpolating cell on the 2D grid
  // Also transform x onto the parametric space [0, 1] x [0, 1]
  const auto [m1, xi, dxi_dx1] = parametric_coordinates(this->_X1, Scalar(this->_x1));
  const auto [m2, eta, deta_dx2] = parametric_coordinates(this->_X2, Scalar(this->_x2));
  auto m = Scalar(at::logical_and(m1.unsqueeze(-1), m2.unsqueeze(-2)));

  // Get the four corner values of the interpolating cell
  //
  // Y01 ------ Y11
  //  |          |
  //  |          |
  //  |          |
  // Y00 ------ Y10
  auto Y00 = this->_Y.batch_index({Ellipsis, Slice(None, -1), Slice(None, -1)});
  auto Y01 = this->_Y.batch_index({Ellipsis, Slice(None, -1), Slice(1)});
  auto Y10 = this->_Y.batch_index({Ellipsis, Slice(1), Slice(None, -1)});
  auto Y11 = this->_Y.batch_index({Ellipsis, Slice(1), Slice(1)});
  Y00 = apply_mask(Y00, m);
  Y01 = apply_mask(Y01, m);
  Y10 = apply_mask(Y10, m);
  Y11 = apply_mask(Y11, m);

  // The interpolation formula is:
  // p = Y00 + c1 * xi + c2 * eta + c3 * xi * eta
  // where c1 = (Y10 - Y00)
  //       c2 = (Y01 - Y00)
  //       c3 = (Y11 - Y10 - Y01 + Y00)
  const auto c1 = Y10 - Y00;
  const auto c2 = Y01 - Y00;
  const auto c3 = Y11 - Y10 - Y01 + Y00;

  if (out)
    this->_p = Y00 + c1 * xi + c2 * eta + c3 * xi * eta;

  if (dout_din)
  {
    if (this->_x1.is_dependent())
      this->_p.d(this->_x1) = (c1 + c3 * eta) * dxi_dx1;
    if (this->_x2.is_dependent())
      this->_p.d(this->_x2) = (c2 + c3 * xi) * deta_dx2;
  }

  if (d2out_din2)
    if (this->_x1.is_dependent() && this->_x2.is_dependent())
    {
      this->_p.d(this->_x1, this->_x2) = c3 * dxi_dx1 * deta_dx2;
      this->_p.d(this->_x2, this->_x1) = c3 * dxi_dx1 * deta_dx2;
    }
}

#define REGISTER(T)                                                                                \
  using T##BilinearInterpolation = BilinearInterpolation<T>;                                       \
  register_NEML2_object(T##BilinearInterpolation);                                                 \
  template class BilinearInterpolation<T>
REGISTER(Scalar);
REGISTER(Vec);
REGISTER(SR2);
} // namespace neml2
