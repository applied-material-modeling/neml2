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

#include "neml2/models/solid_mechanics/crystal_plasticity/DislocationObstacleStrengthMap.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{
register_NEML2_object(DislocationObstacleStrengthMap);

OptionSet
DislocationObstacleStrengthMap::expected_options()
{
  OptionSet options = SlipStrengthMap::expected_options();
  options.doc() =
      "Dislocation density to strength as in Taylor \\f$ \\tau_i = \\tau_{const} + \\alpha \\mu "
      "b \\sqrt{\\rho_i} \\f$.";

  options.add_input("dislocation_density", "Per-slip dislocation density");
  options.add_parameter<Scalar>("constant_strength", "Constant strength offset");
  options.add_parameter<Scalar>("alpha", "Interaction coefficient");
  options.add_parameter<Scalar>("mu", "Shear modulus");
  options.add_parameter<Scalar>("b", "Burgers vector");

  return options;
}

DislocationObstacleStrengthMap::DislocationObstacleStrengthMap(const OptionSet & options)
  : SlipStrengthMap(options),
    _rho(declare_input_variable<Scalar>("dislocation_density")),
    _tau_const(declare_parameter<Scalar>("constant_strength", "constant_strength", true)),
    _alpha(declare_parameter<Scalar>("alpha", "alpha", true)),
    _mu(declare_parameter<Scalar>("mu", "mu", true)),
    _b(declare_parameter<Scalar>("b", "b", true))
{
}

void
DislocationObstacleStrengthMap::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto sqrt_rho = sqrt(_rho());
  const auto coeff = _alpha * _mu * _b;

  if (out)
    _tau = _tau_const + coeff * sqrt_rho;

  if (dout_din)
  {
    _tau.d(_rho) = coeff * 0.5 / sqrt_rho;

    if (const auto * const tau_const = nl_param("constant_strength"))
      _tau.d(*tau_const) = Scalar::ones(_tau.options());

    if (const auto * const alpha = nl_param("alpha"))
      _tau.d(*alpha) = _mu * _b * sqrt_rho;

    if (const auto * const mu = nl_param("mu"))
      _tau.d(*mu) = _alpha * _b * sqrt_rho;

    if (const auto * const b = nl_param("b"))
      _tau.d(*b) = _alpha * _mu * sqrt_rho;
  }
}
} // namespace neml2
