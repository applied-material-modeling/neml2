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

#include "neml2/models/solid_mechanics/PhaseChangeDeformationGradientJacobian.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(PhaseChangeDeformationGradientJacobian);

OptionSet
PhaseChangeDeformationGradientJacobian::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Define the linear isotropic phase change deformation gradient jacobian, "
                  "i.e. \\f$ J = (1+ \\alpha c \\phi^f + (1-c) \\phi^f "
                  "d\\Omega) \\f$, where \\f$ "
                  "\\alpha, d\\Omega \\f$ is the coefficient of phase expansion (CPE) and phase "
                  "change (CPC), \\f$ \\phi^f \\f$ is the fluid "
                  "fraction, "
                  "and \\f$ c \\f$ is the phase fraction.";

  options.set_input("fluid_fraction") = VariableName(STATE, "phi_f");
  options.set("fluid_fraction").doc() = "Volume fraction of the fluid phase.";

  options.set_input("phase_fraction") = VariableName(STATE, "c");
  options.set("phase_fraction").doc() = "Phase fraction during the transformation.";

  options.set_parameter<TensorName<Scalar>>("CPE");
  options.set("CPE").doc() = "Coefficient of phase expansion";

  options.set_parameter<TensorName<Scalar>>("CPC") = {TensorName<Scalar>("0")};
  options.set("CPC").doc() = "Coefficient of phase change";

  options.set<VariableName>("jacobian") = VariableName(STATE, "J");
  options.set("jacobian").doc() = "Phase change deformation gradient jacobian";

  return options;
}

PhaseChangeDeformationGradientJacobian::PhaseChangeDeformationGradientJacobian(
    const OptionSet & options)
  : Model(options),
    _vf(declare_input_variable<Scalar>("fluid_fraction")),
    _c(declare_input_variable<Scalar>("phase_fraction")),
    _alpha(declare_parameter<Scalar>("alpha", "CPE")),
    _dOmega(declare_parameter<Scalar>("dOmega", "CPC")),
    _J(declare_output_variable<Scalar>("jacobian"))
{
}

void
PhaseChangeDeformationGradientJacobian::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _J = (1.0 + _alpha * _c * _vf + (1 - _c) * _vf * _dOmega);
  }

  if (dout_din)
  {
    if (_vf.is_dependent())
    {
      _J.d(_vf) = (_alpha * _c + (1 - _c) * _dOmega);
    }

    if (_c.is_dependent())
    {
      _J.d(_c) = (_alpha * _vf - _vf * _dOmega);
    }
  }
}
} // namespace neml2
