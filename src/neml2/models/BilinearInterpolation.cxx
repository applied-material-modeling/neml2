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

  options.set<Size>("dim") = 0;
  options.set("dim").doc() = "Intermediate dimension along which to interpolate";

  return options;
}

template <typename T>
BilinearInterpolation<T>::BilinearInterpolation(const OptionSet & options)
  : Interpolation<T>(options),
    _X1(this->template declare_parameter<Scalar>("X1", "abscissa1")),
    _X2(this->template declare_parameter<Scalar>("X2", "abscissa2")),
    _x1(this->template declare_input_variable<Scalar>("argument1")),
    _x2(this->template declare_input_variable<Scalar>("argument2")),
    _dim(options.get<Size>("dim"))
{
}

template <typename T>
void
BilinearInterpolation<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  using namespace indexing;

  neml_assert_dbg(this->_X1.intmd_size(_dim) == this->_Y.intmd_size(_dim),
                  "Abscissa 1 has size ",
                  this->_X1.intmd_size(_dim),
                  " along the interpolating dimension ",
                  _dim,
                  ", but the ordinate has size ",
                  this->_Y.intmd_size(_dim),
                  " along the interpolating dimension ",
                  _dim,
                  ".");

  neml_assert_dbg(this->_X2.intmd_size(_dim) == this->_Y.intmd_size(_dim + 1),
                  "Abscissa 2 has size ",
                  this->_X2.intmd_size(_dim),
                  " along the interpolating dimension ",
                  _dim,
                  ", but the ordinate has size ",
                  this->_Y.intmd_size(_dim + 1),
                  " along the interpolating dimension ",
                  _dim + 1,
                  ".");

  // First move the interpolating dimension to the end
  const auto X1 = this->_X1.intmd_movedim(_dim, -1);
  const auto X2 = this->_X2.intmd_movedim(_dim, -1);
  const auto Y = this->_Y.intmd_movedim(_dim, -1).intmd_movedim(_dim, -1);

  // Unsqueeze one intermediate dimension in x to match X and Y
  const auto x1 = this->_x1().intmd_unsqueeze(-1);
  const auto x2 = this->_x2().intmd_unsqueeze(-1);

  // Get masks for the interpolating cell on the 2D grid
  // Also transform x onto the parametric space (0, 1] x (0, 1]
  const auto [m1, xi, dxi_dx1] = parametric_coordinates(X1, x1);
  const auto [m2, eta, deta_dx2] = parametric_coordinates(X2, x2);
  const auto m = m1.intmd_unsqueeze(-1) && m2.intmd_unsqueeze(-2);

  // Get the four corner values of the interpolating cell
  //
  // Y01 ------ Y11
  //  |          |
  //  |          |
  //  |          |
  // Y00 ------ Y10
  auto Y00 = Y.intmd_index({Ellipsis, Slice(None, -1), Slice(None, -1)});
  auto Y01 = Y.intmd_index({Ellipsis, Slice(None, -1), Slice(1)});
  auto Y10 = Y.intmd_index({Ellipsis, Slice(1), Slice(None, -1)});
  auto Y11 = Y.intmd_index({Ellipsis, Slice(1), Slice(1)});
  Y00 = apply_mask(Y00, m, 2);
  Y01 = apply_mask(Y01, m, 2);
  Y10 = apply_mask(Y10, m, 2);
  Y11 = apply_mask(Y11, m, 2);

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
