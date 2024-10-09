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

#include "neml2/models/solid_mechanics/crystal_plasticity/ElasticStrainRate.h"

#include "neml2/tensors/tensors.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ElasticStrainRate);

OptionSet
ElasticStrainRate::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() = "Calculates the elastic strain rate as \\f$\\dot{\\varepsilon} = d - d^p - "
                  "\\varepsilon w + w \\varepsilon \\f$ "
                  "where \\f$ d \\f$ is the deformation rate, \\f$ d^p \\f$ is the plastic "
                  "deformation rate, \\f$ w \\f$ is the vorticity, and \\f$ \\varepsilon \\f$ is "
                  "the elastic strain.";

  options.set_output("elastic_strain_rate") = VariableName("state", "elastic_strain_rate");
  options.set("elastic_strain_rate").doc() = "Name of the elastic strain rate";

  options.set_input("elastic_strain") = VariableName("state", "elastic_strain");
  options.set("elastic_strain").doc() = "Name of the elastic strain";

  options.set_input("deformation_rate") = VariableName("forces", "deformation_rate");
  options.set("deformation_rate").doc() = "Name of the deformation rate";

  options.set_input("vorticity") = VariableName("forces", "vorticity");
  options.set("vorticity").doc() = "Name of the vorticity";

  options.set_input("plastic_deformation_rate") =
      VariableName("state", "internal", "plastic_deformation_rate");
  options.set("plastic_deformation_rate").doc() = "Name of the plastic deformation rate";

  return options;
}

ElasticStrainRate::ElasticStrainRate(const OptionSet & options)
  : Model(options),
    _e_dot(declare_output_variable<SR2>("elastic_strain_rate")),
    _e(declare_input_variable<SR2>("elastic_strain")),
    _d(declare_input_variable<SR2>("deformation_rate")),
    _w(declare_input_variable<WR2>("vorticity")),
    _dp(declare_input_variable<SR2>("plastic_deformation_rate"))
{
}

void
ElasticStrainRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
    _e_dot = _d - _dp + math::skew_and_sym_to_sym(SR2(_e), WR2(_w));

  if (dout_din)
  {
    const auto I = SSR4::identity_sym(_d.options());

    if (_e.is_dependent())
      _e_dot.d(_e) = math::d_skew_and_sym_to_sym_d_sym(WR2(_w));

    if (_d.is_dependent())
      _e_dot.d(_d) = I;

    if (_w.is_dependent())
      _e_dot.d(_w) = math::d_skew_and_sym_to_sym_d_skew(SR2(_e));

    if (_dp.is_dependent())
      _e_dot.d(_dp) = -I;
  }
}
} // namespace neml2
