// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/models/solid_mechanics/crystal_plasticity/PlasticDeformationRate.h"
#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/list_tensors.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(PlasticDeformationRate);

OptionSet
PlasticDeformationRate::expected_options()
{
  OptionSet options = Model::expected_options();

  options.set<LabeledAxisAccessor>("plastic_deformation_rate") =
      vecstr{"state", "internal", "plastic_deformation_rate"};

  options.set<LabeledAxisAccessor>("orientation") = vecstr{"state", "orientation"};

  options.set<LabeledAxisAccessor>("slip_rates") = vecstr{"state", "internal", "slip_rates"};

  options.set<std::string>("crystal_geometry_name") = "crystal_geometry";

  return options;
}

PlasticDeformationRate::PlasticDeformationRate(const OptionSet & options)
  : Model(options),
    plastic_deformation_rate(
        declare_output_variable<SR2>(options.get<LabeledAxisAccessor>("plastic_deformation_rate"))),
    orientation(declare_input_variable<Rot>(options.get<LabeledAxisAccessor>("orientation"))),
    crystal_geometry(include_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry_name"))),
    slip_rates(declare_input_variable_list<Scalar>(options.get<LabeledAxisAccessor>("slip_rates"),
                                                   crystal_geometry.nslip()))
{
  setup();
}

void
PlasticDeformationRate::set_value(const LabeledVector & in,
                                  LabeledVector * out,
                                  LabeledMatrix * dout_din,
                                  LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the input
  const auto R = in.get<Rot>(orientation);
  const auto g = in.get_list<Scalar>(slip_rates);
  auto dp_crystal = (g * crystal_geometry.M()).list_sum();

  if (out)
    out->set(dp_crystal.rotate(R), plastic_deformation_rate);

  if (dout_din)
  {
    dout_din->set(list_derivative_outer_product_b(
                      [](auto a, auto b) { return b.rotate(a); }, R, crystal_geometry.M()),
                  plastic_deformation_rate,
                  slip_rates);
    dout_din->set(dp_crystal.drotate(R), plastic_deformation_rate, orientation);
  }
}
} // namespace neml2
