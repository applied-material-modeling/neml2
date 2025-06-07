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

#include "neml2/models/solid_mechanics/ThermalDeformationGradientJacobian.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(ThermalDeformationGradientJacobian);

OptionSet
ThermalDeformationGradientJacobian::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Define the  linear isotropic thermal deformation gradient jacobian, "
      "i.e. \\f$ J = (1+ \\alpha (T - T_0))\\f$, where \\f$ "
      "\\alpha \\f$ is the coefficient of thermal expansion (CTE), \\f$ T \\f$ is the temperature, "
      "and \\f$ T_0 \\f$ is the reference (stress-free) temperature.";

  options.set_input("temperature") = VariableName(FORCES, "T");
  options.set("temperature").doc() = "Temperature";

  options.set_buffer<TensorName<Scalar>>("reference_temperature");
  options.set("reference_temperature").doc() = "Reference (stress-free) temperature";

  options.set_parameter<TensorName<Scalar>>("CTE");
  options.set("CTE").doc() = "Coefficient of thermal expansion";

  options.set<VariableName>("jacobian") = VariableName(STATE, "J");
  options.set("jacobian").doc() = "Tempearture deformation gradient jacobian";

  return options;
}

ThermalDeformationGradientJacobian::ThermalDeformationGradientJacobian(const OptionSet & options)
  : Model(options),
    _T(declare_input_variable<Scalar>("temperature")),
    _T0(declare_buffer<Scalar>("T0", "reference_temperature")),
    _alpha(declare_parameter<Scalar>("alpha", "CTE", true)),
    _J(declare_output_variable<Scalar>("jacobian"))
{
}

void
ThermalDeformationGradientJacobian::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _J = 1.0 + _alpha * (_T - _T0);
  }

  if (dout_din)
  {
    if (_T.is_dependent())
    {
      _J.d(_T) = _alpha;
    }

    if (const auto * const alpha = nl_param("alpha"))
    {
      _J.d(*alpha) = (_T - _T0);
    }
  }
}
} // namespace neml2
