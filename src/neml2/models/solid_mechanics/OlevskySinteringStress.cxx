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

#include "neml2/models/solid_mechanics/OlevskySinteringStress.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(OlevskySinteringStress);

OptionSet
OlevskySinteringStress::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Define the Olevsky-Skorohod sintering stress to be used in conjunction with "
                  "poroplasticity yield functions such as the GTNYieldFunction. The sintering "
                  "stress is defined as \\f$ \\sigma_s = 3 \\dfrac{\\gamma}{r} \\phi^2 \\f$, where "
                  "\\f$ \\gamma \\f$ is the surface tension, \\f$ r \\f$ is the size of the "
                  "particles/powders, and \\f$ \\phi \\f$ is the void fraction.";

  options.set<bool>("define_second_derivatives") = true;

  options.set_output("sintering_stress") = VariableName(STATE, "internal", "ss");
  options.set("sintering_stress").doc() = "Sintering stress";

  options.set_input("void_fraction") = VariableName(STATE, "internal", "f");
  options.set("void_fraction").doc() = "Void fraction";

  options.set_parameter<TensorName<Scalar>>("surface_tension");
  options.set("surface_tension").doc() = "Surface tension";

  options.set_parameter<TensorName<Scalar>>("particle_radius");
  options.set("particle_radius").doc() = "Particle radius";

  return options;
}

OlevskySinteringStress::OlevskySinteringStress(const OptionSet & options)
  : Model(options),
    _s(declare_output_variable<Scalar>("sintering_stress")),
    _phi(declare_input_variable<Scalar>("void_fraction")),
    _gamma(declare_parameter<Scalar>("gamma", "surface_tension")),
    _r(declare_parameter<Scalar>("r", "particle_radius"))
{
}

void
OlevskySinteringStress::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _s = 3 * _gamma * _phi * _phi / _r;

  if (dout_din)
    if (_phi.is_dependent())
      _s.d(_phi) = 6 * _gamma * _phi / _r;

  if (d2out_din2)
    if (_phi.is_dependent())
      _s.d2(_phi, _phi) = 6 * _gamma / _r;
}
} // namespace neml2
