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

#include "neml2/models/solid_mechanics/crystal_plasticity/PlasticDeformationRate.h"
#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/list_tensors.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(PlasticDeformationRate);

OptionSet
PlasticDeformationRate::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() = "Caclulates the plastic deformation rate as \\f$ d^p = \\sum_{i=1}^{n_{slip}} "
                  "\\dot{\\gamma}_i Q \\operatorname{sym}{\\left(d_i \\otimes n_i \\right)} Q^T "
                  "\\f$ with \\f$ d^p \\f$ the plastic deformation rate, \\f$ \\dot{\\gamma}_i "
                  "\\f$ the slip rate on the ith slip system, \\f$Q \\f$ the orientation, \\f$ d_i "
                  "\\f$ the slip system direction, and \\f$ n_i \\f$ the slip system normal.";

  options.set_output("plastic_deformation_rate") =
      VariableName("state", "internal", "plastic_deformation_rate");
  options.set("plastic_deformation_rate").doc() = "The name of the plastic deformation rate tensor";

  options.set_input("orientation") = VariableName("state", "orientation_matrix");
  options.set("orientation").doc() = "The name of the orientation matrix tensor";

  options.set_input("slip_rates") = VariableName("state", "internal", "slip_rates");
  options.set("slip_rates").doc() = "The name of the tensor containg the current slip rates";

  options.set<std::string>("crystal_geometry_name") = "crystal_geometry";
  options.set("crystal_geometry_name").doc() =
      "The name of the Data object containing the crystallographic information for the material";

  return options;
}

PlasticDeformationRate::PlasticDeformationRate(const OptionSet & options)
  : Model(options),
    _crystal_geometry(register_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry_name"))),
    _dp(declare_output_variable<SR2>("plastic_deformation_rate")),
    _R(declare_input_variable<R2>("orientation")),
    _g(declare_input_variable_list<Scalar>(_crystal_geometry.nslip(), "slip_rates"))
{
}

void
PlasticDeformationRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the input
  const auto g = Scalar(_g, batch_dim() + 1);
  const auto dp_crystal = math::batch_sum(g * _crystal_geometry.M(), -1);

  if (out)
    _dp = dp_crystal.rotate(R2(_R));

  if (dout_din)
  {
    if (_g.is_dependent())
      _dp.d(_g) = Tensor(_crystal_geometry.M().rotate(R2(_R).batch_unsqueeze(-1)), batch_dim())
                      .base_transpose(-2, -1);

    if (_R.is_dependent())
      _dp.d(_R) = dp_crystal.drotate(R2(_R));
  }
}
} // namespace neml2
