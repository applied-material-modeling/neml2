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

#include "neml2/models/Determinant.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/det.h"
#include "neml2/tensors/functions/inv.h"

namespace neml2
{
using R2Determinant = Determinant<R2>;
using SR2Determinant = Determinant<SR2>;
register_NEML2_object(R2Determinant);
register_NEML2_object(SR2Determinant);

template <typename T>
OptionSet
Determinant<T>::expected_options()
{
  auto options = Model::expected_options();
  options.doc() = "Calculate the Jacobian of a second order tensor.";

  options.set_input("input") = VariableName(STATE, "F");
  options.set("input").doc() = "The second order tensor to calculate the determinant of";

  options.set_output("determinant") = VariableName(STATE, "J");
  options.set("determinant").doc() = "The determinant of the input tensor";

  return options;
}

template <typename T>
Determinant<T>::Determinant(const OptionSet & options)
  : Model(options),
    _F(declare_input_variable<T>("input")),
    _J(declare_output_variable<Scalar>("determinant"))
{
}

template <typename T>
void
Determinant<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _J = neml2::det(_F());

  if (dout_din)
    if (_F.is_dependent())
      _J.d(_F) = neml2::det(_F()) * neml2::inv(_F()).transpose();
}

template class Determinant<R2>;
template class Determinant<SR2>;
} // namespace neml2
