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

#include "neml2/models/solid_mechanics/SwellingAndPhaseChangeDeformationJacobian.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(SwellingAndPhaseChangeDeformationJacobian);

OptionSet
SwellingAndPhaseChangeDeformationJacobian::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Define the linear isotropic phase change deformation Jacobian for a freezing liquid or a "
      "melting solid, i.e. \\f$ J = \\left( 1 + \\alpha c \\phi^f + (1-c) \\phi^f \\Delta \\Omega "
      "\\right) \\f$, where \\f$ \\alpha \\f$ is the coefficient of swelling, \\f$ \\Delta \\Omega "
      "\\f$ is relative difference of the reference volume between the two phases, \\f$ \\phi^f "
      "\\f$ "
      "is the fluid fraction associated with swelling, and \\f$ c \\f$ is the phase fraction.";

  options.add_input("fluid_fraction", "Volume fraction of the fluid phase.");
  options.add_output("jacobian", "Phase change deformation Jacobian");

  options.add_parameter<Scalar>(
      "phase_fraction",
      "Phase fraction during the phase change. 0 means all solid, 1 means all liquid.");
  options.add_parameter<Scalar>("swelling_coefficient", "Coefficient of phase expansion");
  options.add_parameter<Scalar>(
      "reference_volume_difference",
      "Relative difference between the reference volumes of the two phases.");

  return options;
}

SwellingAndPhaseChangeDeformationJacobian::SwellingAndPhaseChangeDeformationJacobian(
    const OptionSet & options)
  : Model(options),
    _vf(declare_input_variable<Scalar>("fluid_fraction")),
    _c(declare_parameter<Scalar>("c", "phase_fraction", /*allow_nonlinear=*/true)),
    _alpha(declare_parameter<Scalar>("alpha", "swelling_coefficient")),
    _dOmega(declare_parameter<Scalar>("dOmega", "reference_volume_difference")),
    _J(declare_output_variable<Scalar>("jacobian"))
{
}

void
SwellingAndPhaseChangeDeformationJacobian::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _J = (1.0 + _alpha * _c * _vf + (1 - _c) * _vf * _dOmega);

  if (dout_din)
  {
    _J.d(_vf) = (_alpha * _c + (1 - _c) * _dOmega);

    if (const auto * const c = nl_param("c"))
      _J.d(*c) = (_alpha * _vf - _vf * _dOmega);
  }
}
} // namespace neml2
