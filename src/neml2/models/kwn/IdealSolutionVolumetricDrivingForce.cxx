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

#include "neml2/models/kwn/IdealSolutionVolumetricDrivingForce.h"

#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/log.h"

namespace neml2
{
register_NEML2_object(IdealSolutionVolumetricDrivingForce);

OptionSet
IdealSolutionVolumetricDrivingForce::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Compute the molar Gibbs free energy of precipitation under the ideal-solution "
      "approximation, \\f$ \\Delta g = R T \\sum_k w_k \\ln(c_k / c_k^{\\mathrm{eq}}) "
      "\\f$. For a compound precipitate, supplying every participating component with weight 1 "
      "reproduces the \"product of all components\" rule used by Hu and Cocks "
      "([MSEA 772, 138787, Eq. 9](https://doi.org/10.1016/j.msea.2019.138787)); "
      "the V_m factor that appears in the per-volume form is absorbed into the V_m^2 term of "
      "`NucleationBarrierAndCriticalRadius`, so this object outputs J/mol directly.";

  options.add_input("temperature", "Temperature");

  options.add<std::vector<VariableName>>(
      "current_concentrations",
      "Current matrix concentrations for each species participating in the precipitate");

  options.add<std::vector<TensorName<Scalar>>>(
      "equilibrium_concentrations",
      "Equilibrium matrix concentrations for each species. May be supplied as constants or as "
      "outputs of upstream models (e.g. temperature-dependent interpolations).");

  options.add<std::vector<TensorName<Scalar>>>(
      "weights",
      {TensorName<Scalar>("1")},
      "Stoichiometric weights for each species. If a single entry is supplied, the same weight is "
      "used for every species. Default is unit weights for every species, matching the "
      "Hu--Cocks compound-precipitate convention.");

  options.add_parameter<Scalar>("gas_constant", "Universal gas constant");

  options.add_output("driving_force", "Molar Gibbs free energy of precipitation");

  return options;
}

IdealSolutionVolumetricDrivingForce::IdealSolutionVolumetricDrivingForce(const OptionSet & options)
  : Model(options),
    _T(declare_input_variable<Scalar>("temperature")),
    _R_g(declare_parameter<Scalar>("R_g", "gas_constant")),
    _dg(declare_output_variable<Scalar>("driving_force"))
{
  const auto x_names = options.get<std::vector<VariableName>>("current_concentrations");
  const auto x_eq_refs = options.get<std::vector<TensorName<Scalar>>>("equilibrium_concentrations");
  auto w_refs = options.get<std::vector<TensorName<Scalar>>>("weights");

  neml_assert(!x_names.empty(),
              "IdealSolutionVolumetricDrivingForce requires at least one species; got an empty "
              "current_concentrations list.");

  neml_assert(x_eq_refs.size() == x_names.size(),
              "Number of equilibrium_concentrations (",
              x_eq_refs.size(),
              ") does not match number of current_concentrations (",
              x_names.size(),
              ").");

  neml_assert(w_refs.size() == 1 || w_refs.size() == x_names.size(),
              "Expected 1 or ",
              x_names.size(),
              " entries in weights, got ",
              w_refs.size(),
              ".");

  // Broadcast a single weight to all species
  if (w_refs.size() == 1 && x_names.size() > 1)
    w_refs = std::vector<TensorName<Scalar>>(x_names.size(), w_refs[0]);

  _xs.reserve(x_names.size());
  _x_eqs.reserve(x_names.size());
  _ws.reserve(x_names.size());
  for (std::size_t i = 0; i < x_names.size(); i++)
  {
    _xs.push_back(&declare_input_variable<Scalar>(x_names[i]));
    _x_eqs.push_back(&declare_parameter<Scalar>(
        "x_eq_" + std::to_string(i), x_eq_refs[i], /*allow_nonlinear=*/true));
    _ws.push_back(&declare_parameter<Scalar>("w_" + std::to_string(i), w_refs[i], false));
  }
}

void
IdealSolutionVolumetricDrivingForce::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto T = _T();
  const auto Rg = _R_g;
  const auto pref = Rg * T;

  // sum_log = sum_k w_k * log(c_k / c_k_eq)
  auto sum_log = Scalar::zeros_like((*_xs[0])());
  for (std::size_t i = 0; i < _xs.size(); i++)
    sum_log = sum_log + (*_ws[i]) * neml2::log((*_xs[i])() / (*_x_eqs[i]));

  if (out)
    _dg = pref * sum_log;

  if (dout_din)
  {
    // d(dg)/d(T) = R * sum_log
    _dg.d(_T) = Rg * sum_log;

    for (std::size_t i = 0; i < _xs.size(); i++)
    {
      // d(dg)/d(c_k) = R T * w_k / c_k
      _dg.d(*_xs[i]) = pref * (*_ws[i]) / (*_xs[i])();

      if (const auto * const x_eq_param = nl_param("x_eq_" + std::to_string(i)))
        _dg.d(*x_eq_param) = -pref * (*_ws[i]) / (*_x_eqs[i]);
    }
  }
}
} // namespace neml2
