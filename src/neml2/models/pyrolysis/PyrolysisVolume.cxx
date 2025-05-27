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

#include "neml2/models/pyrolysis/PyrolysisVolume.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(PyrolysisVolume);
OptionSet
PyrolysisVolume::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the volume from the pyrolysis process of a composite systems consisted of binder, "
      "solid, particles, gas in closed pores and open pores. The volume has the form of \\f$ "
      "\\frac{\\frac{M\\omega_b}{\\rho_b} "
      "+\\frac{M\\omega_p}{\\rho_p}+\\frac{M\\omega_s}{\\rho_s}+\\frac{M\\omega_{cp}}{\\rho_{cp}}"
      "}{1-\\phi_{op}} "
      "where \\f$ \\omega_i, \\rho_i (i = b,p,s,cp) \\f$ is the mass fraction and density of "
      "binder, "
      "particle, solid "
      "and closed pore gas, and "
      "\\f$ \\phi_{op} \\f$ is the volume fraction of the open pores - which assumed to have no "
      "gas. \\f$ M \\f is the reference mass of the composite.";

  options.set_parameter<TensorName<Scalar>>("density_binder");
  options.set("density_binder").doc() = "Density of the binder";

  options.set_parameter<TensorName<Scalar>>("density_solid");
  options.set("density_solid").doc() = "Density of the solid";

  options.set_parameter<TensorName<Scalar>>("density_particle");
  options.set("density_particle").doc() = "Density of the particle";

  options.set_parameter<TensorName<Scalar>>("density_closed_pore_gas");
  options.set("density_closed_pore_gas").doc() = "Density of the gas in the closed pores";

  options.set_parameter<TensorName<Scalar>>("reference_mass");
  options.set("reference_mass").doc() = "Reference mass of the composite";

  options.set_input("binder_mass_fraction") = VariableName(STATE, "binder_mass_fraction");
  options.set("binder_mass_fraction").doc() = "Mass fraction of the binder";

  options.set_input("solid_mass_fraction") = VariableName(STATE, "solid_mass_fraction");
  options.set("solid_mass_fraction").doc() = "Mass fraction of the solid";

  options.set_input("particle_mass_fraction") = VariableName(STATE, "particle_mass_fraction");
  options.set("particle_mass_fraction").doc() = "Mass fraction of the particle";

  options.set_input("close_pore_gas_mass_fraction") =
      VariableName(STATE, "close_pore_gas_mass_fraction");
  options.set("close_pore_gas_mass_fraction").doc() = "Mass fraction of the close-pore gas";

  options.set_input("open_pore_volume_fraction") = VariableName(STATE, "open_pore_volume_fraction");
  options.set("open_pore_volume_fraction").doc() = "Volume fraction of the open pore";

  options.set_output("pyrolysis_composite_volume") = VariableName(STATE, "out");
  options.set("pyrolysis_composite_volume").doc() =
      "Volume of the composite from the pyrolysis process.";

  return options;
}

PyrolysisVolume::PyrolysisVolume(const OptionSet & options)
  : Model(options),
    _rhob(declare_parameter<Scalar>("rho_b", "density_binder")),
    _rhos(declare_parameter<Scalar>("rho_s", "density_solid")),
    _rhop(declare_parameter<Scalar>("rho_p", "density_particle")),
    _rhog(declare_parameter<Scalar>("rho_g", "density_closed_pore_gas")),
    _M(declare_parameter<Scalar>("M", "reference_mass")),
    _wb(declare_input_variable<Scalar>("binder_mass_fraction")),
    _ws(declare_input_variable<Scalar>("solid_mass_fraction")),
    _wp(declare_input_variable<Scalar>("particle_mass_fraction")),
    _wg(declare_input_variable<Scalar>("close_pore_gas_mass_fraction")),
    _phiop(declare_input_variable<Scalar>("open_pore_volume_fraction")),
    _V(declare_output_variable<Scalar>("pyrolysis_composite_volume"))
{
}

void
PyrolysisVolume::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto V = _M * (_wb / _rhob + _ws / _rhos + _wp / _rhop + _wg / _rhog) / (1 - _phiop);
  if (out)
  {
    _V = V;
  }

  if (dout_din)
  {
    _V.d(_wb) = _M / (_rhob * (1 - _phiop));
    _V.d(_ws) = _M / (_rhos * (1 - _phiop));
    _V.d(_wp) = _M / (_rhop * (1 - _phiop));
    _V.d(_wg) = _M / (_rhog * (1 - _phiop));
    _V.d(_phiop) = V / (1 - _phiop);
  }
}
}