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

#include "neml2/models/solid_mechanics/crystal_plasticity/PerSlipForestDislocationEvolution.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/sign.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{
register_NEML2_object(PerSlipForestDislocationEvolution);

OptionSet
PerSlipForestDislocationEvolution::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() =
      "Standard forest hardening model per slip system defined by \\f$ \\dot{\\rho}_i = "
      "(k_1 \\sqrt{\\rho_i} - k_2 \\rho_i) \\left| \\dot{\\gamma}_i \\right| \\f$.";

  options.set_output("dislocation_density_rate") =
      VariableName(STATE, "internal", "dislocation_density_rate");
  options.set("dislocation_density_rate").doc() = "Per-slip dislocation density rate";

  options.set_input("dislocation_density") = VariableName(STATE, "internal", "dislocation_density");
  options.set("dislocation_density").doc() = "Per-slip dislocation density";

  options.set_input("slip_rates") = VariableName(STATE, "internal", "slip_rates");
  options.set("slip_rates").doc() = "Per-slip system slip rates";

  options.set_parameter<TensorName<Scalar>>("k1");
  options.set("k1").doc() = "Hardening coefficient";

  options.set_parameter<TensorName<Scalar>>("k2");
  options.set("k2").doc() = "Recovery coefficient";

  return options;
}

PerSlipForestDislocationEvolution::PerSlipForestDislocationEvolution(const OptionSet & options)
  : Model(options),
    _rho_dot(declare_output_variable<Scalar>("dislocation_density_rate")),
    _rho(declare_input_variable<Scalar>("dislocation_density")),
    _gamma_dot(declare_input_variable<Scalar>("slip_rates")),
    _k1(declare_parameter<Scalar>("k1", "k1", true)),
    _k2(declare_parameter<Scalar>("k2", "k2", true))
{
}

void
PerSlipForestDislocationEvolution::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto sqrt_rho = sqrt(_rho());
  const auto abs_gamma = abs(_gamma_dot());
  const auto hardening = _k1 * sqrt_rho - _k2 * _rho();

  if (out)
    _rho_dot = hardening * abs_gamma;

  if (dout_din)
  {
    if (_rho.is_dependent())
      _rho_dot.d(_rho) = (_k1 * 0.5 / sqrt_rho - _k2) * abs_gamma;

    if (_gamma_dot.is_dependent())
      _rho_dot.d(_gamma_dot) = hardening * sign(_gamma_dot());

    if (const auto * const k1 = nl_param("k1"))
      _rho_dot.d(*k1) = sqrt_rho * abs_gamma;

    if (const auto * const k2 = nl_param("k2"))
      _rho_dot.d(*k2) = -_rho() * abs_gamma;
  }
}
} // namespace neml2
