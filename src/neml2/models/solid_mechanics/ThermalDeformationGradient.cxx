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

#include "neml2/models/solid_mechanics/ThermalDeformationGradient.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(ThermalDeformationGradient);

OptionSet
ThermalDeformationGradient::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Define the  linear isotropic thermal deformation gradient, "
      "i.e. \\f$ \\boldsymbol{F}_T = (1+ \\alpha (T - T_0))^{1/3} \\boldsymbol{I} \\f$, where \\f$ "
      "\\alpha \\f$ is the coefficient of thermal expansion (CTE), \\f$ T \\f$ is the temperature, "
      "and \\f$ T_0 \\f$ is the reference (stress-free) temperature.";

  options.set_input("temperature") = VariableName(FORCES, "T");
  options.set("temperature").doc() = "Temperature";

  options.set_buffer<TensorName<Scalar>>("reference_temperature");
  options.set("reference_temperature").doc() = "Reference (stress-free) temperature";

  options.set_parameter<TensorName<Scalar>>("CTE");
  options.set("CTE").doc() = "Coefficient of thermal expansion";

  options.set<bool>("inverse_condition") = false;
  options.set("inverse_condition").doc() = "Whether to take the inverse operation.";

  options.set<VariableName>("deformation_gradient") = VariableName(STATE, "F");
  options.set("deformation_gradient").doc() = "Tempearture deformation gradient tensor";

  return options;
}

ThermalDeformationGradient::ThermalDeformationGradient(const OptionSet & options)
  : Model(options),
    _T(declare_input_variable<Scalar>("temperature")),
    _T0(declare_buffer<Scalar>("T0", "reference_temperature")),
    _alpha(declare_parameter<Scalar>("alpha", "CTE", true)),
    _inverse(options.get<bool>("inverse_condition")),
    _F(declare_output_variable<R2>("deformation_gradient"))
{
}

void
ThermalDeformationGradient::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto term = (1.0 + _alpha * (_T - _T0));
  if (out)
  {
    _F = pow(term, 1.0 / 3.0) * R2::identity(_T.options());
    if (_inverse)
      _F = pow(term, -1.0 / 3.0) * R2::identity(_T.options());
  }

  if (dout_din)
  {
    if (_T.is_dependent())
    {
      _F.d(_T) = 1.0 / 3.0 * pow(term, -2.0 / 3.0) * _alpha * R2::identity(_T.options());
      if (_inverse)
        _F.d(_T) = -1.0 / 3.0 * pow(term, -4.0 / 3.0) * _alpha * R2::identity(_T.options());
    }

    if (const auto * const alpha = nl_param("alpha"))
    {
      _F.d(*alpha) = 1.0 / 3.0 * pow(term, -2.0 / 3.0) * (_T - _T0) * R2::identity(_T.options());
      if (_inverse)
        _F.d(*alpha) = -1.0 / 3.0 * pow(term, -4.0 / 3.0) * (_T - _T0) * R2::identity(_T.options());
    }
  }
}
} // namespace neml2
