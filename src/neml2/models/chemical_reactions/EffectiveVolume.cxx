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

#include "neml2/models/chemical_reactions/EffectiveVolume.h"

#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(EffectiveVolume);
OptionSet
EffectiveVolume::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the total volume of a control mass. The volume has the form of \\f$ V = "
      "\\dfrac{M}{1-\\phi_{o}} \\sum_i \\frac{\\omega_i}{\\rho_i} \\f$, where \\f$ \\omega_i "
      "\\f$ and \\f$ \\rho_i \\f$ are respectively the mass fraction and the density of each "
      "component; \\f$ \\phi_{o} \\f$ is the volume fraction accounting for leakage from the "
      "control mass; \\f$ M \\f is the reference mass of the composite.";

  options.set_parameter<TensorName<Scalar>>("reference_mass");
  options.set("reference_mass").doc() = "Reference mass of the composite";

  options.set_input("open_volume_fraction");
  options.set("open_volume_fraction").doc() = "Open volume fraction accounting for leakage";

  options.set<std::vector<VariableName>>("mass_fractions");
  options.set("mass_fractions").doc() = "Mass fractions of the components in the composite";

  options.set_parameter<std::vector<TensorName<Scalar>>>("densities");
  options.set("densities").doc() = "Densities of the components in the composite";

  options.set_output("composite_volume") = VariableName(STATE, "V");
  options.set("composite_volume").doc() = "Volume of the composite";

  return options;
}

EffectiveVolume::EffectiveVolume(const OptionSet & options)
  : Model(options),
    _M(declare_parameter<Scalar>("M", "reference_mass")),
    _phio(options.get("open_volume_fraction").user_specified()
              ? &declare_input_variable<Scalar>("open_volume_fraction")
              : nullptr),
    _V(declare_output_variable<Scalar>("composite_volume"))
{
  for (const auto & v : options.get<std::vector<VariableName>>("mass_fractions"))
    _ws.push_back(&declare_input_variable<Scalar>(v));

  // densities
  const auto rho_refs = options.get<std::vector<TensorName<Scalar>>>("densities");

  // sizes must match
  neml_assert(rho_refs.size() == _ws.size(),
              "Number of  mass fractions (",
              _ws.size(),
              ") does not match number of densities (",
              rho_refs.size(),
              ").");

  // declare densities as parameters
  _rhos.resize(_ws.size());
  for (std::size_t i = 0; i < _ws.size(); i++)
    _rhos[i] = &declare_parameter<Scalar>(
        "rho_" + std::to_string(i), rho_refs[i], /*allow_nonlinear=*/false);
}

void
EffectiveVolume::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto sum = Scalar::zeros_like(*_ws[0]);

  for (std::size_t i = 0; i < _ws.size(); i++)
    sum = sum + *_ws[i] / *_rhos[i];

  auto coef = _phio ? _M / (1 - *_phio) : _M;

  if (out)
    _V = coef * sum;

  if (dout_din)
  {
    for (std::size_t i = 0; i < _ws.size(); i++)
      if (_ws[i]->is_dependent())
        _V.d(*_ws[i]) = coef / *_rhos[i];

    if (_phio && _phio->is_dependent())
      _V.d(*_phio) = _M * sum / ((1 - *_phio) * (1 - *_phio));
  }
}
}
