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

#include "neml2/models/porous_flow/AdvectiveStress.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/inner.h"

namespace neml2
{
register_NEML2_object(AdvectiveStress);
OptionSet
AdvectiveStress::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the advective stress, \\f$ p_s \\f$, taking the form of \\f$ p_s = \\frac{c}{3J} "
      "P_{ij}F_{ij} \\f$. Here, \\f$ J, P, F \\f$ are the the deformation gradient Jacobian, the "
      "1st Piola-Kirchhoff stress and the defomration gradient. \\f$ c \\f$ is the volume change "
      "coefficient.";

  options.set_parameter<TensorName<Scalar>>("coefficient");
  options.set("coefficient").doc() = "Coefficient c";

  options.set_input("js");
  options.set("js").doc() =
      "The Jacobian of the deformation gradient associated with the swelling and phase change";

  options.set_input("jt");
  options.set("jt").doc() =
      "The Jacobian of the deformation gradient associated with the thermal and volume expansion";

  options.set_input("deformation_gradient") = VariableName(FORCES, "F");
  options.set("deformation_gradient").doc() = "The deformation gradient";

  options.set_input("pk1_stress") = VariableName(STATE, "P");
  options.set("pk1_stress").doc() = "1st Piola-Kirchhoff stress";

  options.set_output("advective_stress") = VariableName(STATE, "advective_stress");
  options.set("advective_stress").doc() = "The average advective stress";

  return options;
}

AdvectiveStress::AdvectiveStress(const OptionSet & options)
  : Model(options),
    _coeff(declare_parameter<Scalar>("coeff", "coefficient")),
    _Js(options.get("js").user_specified() ? &declare_input_variable<Scalar>("js") : nullptr),
    _Jt(options.get("jt").user_specified() ? &declare_input_variable<Scalar>("jt") : nullptr),
    _P(declare_input_variable<R2>("pk1_stress")),
    _F(declare_input_variable<R2>("deformation_gradient")),
    _ps(declare_output_variable<Scalar>("advective_stress"))
{
}

void
AdvectiveStress::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  Scalar Js = _Js ? (*_Js)() : Scalar::full(1.0, _P.options());
  Scalar Jt = _Jt ? (*_Jt)() : Scalar::full(1.0, _P.options());

  if (out)
  {
    _ps = -_coeff / 3.0 * pow(Js, -5.0 / 3.0) * pow(Jt, -2.0 / 3.0) * neml2::inner(_P(), _F());
  }

  if (dout_din)
  {
    const auto I = R2::identity(_P.options());

    if (_Js && _Js->is_dependent())
      _ps.d(*_Js) = 5.0 / 3.0 * _coeff / 3.0 * pow(Js, -8.0 / 3.0) * pow(Jt, -2.0 / 3.0) *
                    neml2::inner(_P(), _F());

    if (_Jt && _Jt->is_dependent())
      _ps.d(*_Jt) = 2.0 / 3.0 * _coeff / 3.0 * pow(Js, -5.0 / 3.0) * pow(Jt, -5.0 / 3.0) *
                    neml2::inner(_P(), _F());

    if (_P.is_dependent())
      _ps.d(_P) = -_coeff / 3.0 * pow(Js, -5.0 / 3.0) * pow(Jt, -2.0 / 3.0) * _F;

    if (_F.is_dependent())
      _ps.d(_F) = -_coeff / 3.0 * pow(Js, -5.0 / 3.0) * pow(Jt, -2.0 / 3.0) * _P;
  }
}
} // namespace neml2
