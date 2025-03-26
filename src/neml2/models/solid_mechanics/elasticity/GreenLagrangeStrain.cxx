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

#include "neml2/models/solid_mechanics/elasticity/GreenLagrangeStrain.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/mandel_notation.h"

namespace neml2
{
register_NEML2_object(GreenLagrangeStrain);

OptionSet
GreenLagrangeStrain::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Green-Lagrange strain, \\f$ E = \\frac{1}{2} (C - I) \\f$, where \\f$ C = F^T F \\f$ "
      "is the right Cauchy-Green tensor and \\f$ I \\f$ is the identity tensor.";

  options.set_input("deformation_gradient") = VariableName(FORCES, "F");
  options.set("deformation_gradient").doc() = "The deformation gradient";

  options.set_output("strain") = VariableName(STATE, "E");
  options.set("strain").doc() = "The Green-Lagrange strain";

  return options;
}

GreenLagrangeStrain::GreenLagrangeStrain(const OptionSet & options)
  : Model(options),
    _E(declare_output_variable<SR2>("strain")),
    _F(declare_input_variable<R2>("deformation_gradient"))
{
}

void
GreenLagrangeStrain::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
  {
    const auto C = R2(_F).transpose() * _F;
    _E = 0.5 * (SR2(C) - SR2::identity(_F.options()));
  }

  if (dout_din)
  {
    const auto I = R2::identity(_F.options());
    const auto dC_dF = R4(at::einsum("...jm,...ni,...jk", {I, I, _F})) +
                       R4(at::einsum("...jm,...nk,...ji", {I, I, _F}));
    _E.d(_F) = 0.5 * full_to_mandel(dC_dF);
  }
}

} // namespace neml2
