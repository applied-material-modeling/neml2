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

#include "neml2/models/solid_mechanics/crystal_plasticity/PlasticSpatialVelocityGradient.h"
#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SFFR4.h"
#include "neml2/tensors/functions/sum.h"

namespace neml2
{
register_NEML2_object(PlasticSpatialVelocityGradient);

OptionSet
PlasticSpatialVelocityGradient::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() =
      "Caclulates the plastic spatial velocity gradient as \\f$ l^p = \\sum_{i=1}^{n_{slip}} "
      "\\dot{\\gamma}_i Q \\left(d_i \\otimes n_i \\right) Q^T "
      "\\f$ with \\f$ l^p \\f$ the plastic spatial velocity gradient, \\f$ \\dot{\\gamma}_i "
      "\\f$ the slip rate on the ith slip system, \\f$Q \\f$ the orientation, \\f$ d_i "
      "\\f$ the slip system direction, and \\f$ n_i \\f$ the slip system normal.";

  options.set_output("plastic_spatial_velocity_gradient") =
      VariableName(STATE, "internal", "plastic_spatial_velocity_gradient");
  options.set("plastic_spatial_velocity_gradient").doc() =
      "The name of the plastic spatial velocity gradient";

  options.set_input("orientation") = VariableName(STATE, "orientation_matrix");
  options.set("orientation").doc() = "The name of the orientation matrix tensor";

  options.set_input("slip_rates") = VariableName(STATE, "internal", "slip_rates");
  options.set("slip_rates").doc() = "The name of the tensor containg the current slip rates";

  options.set<std::string>("crystal_geometry") = "crystal_geometry";
  options.set("crystal_geometry").doc() =
      "The name of the Data object containing the crystallographic information for the material";

  return options;
}

PlasticSpatialVelocityGradient::PlasticSpatialVelocityGradient(const OptionSet & options)
  : Model(options),
    _crystal_geometry(register_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry"))),
    _lp(declare_output_variable<R2>("plastic_spatial_velocity_gradient")),
    _R(declare_input_variable<R2>("orientation")),
    _g(declare_input_variable<Scalar>("slip_rates"))
{
}

void
PlasticSpatialVelocityGradient::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto A = _crystal_geometry.A();
  const auto lp_crystal = intmd_sum(_g * A, -1, /*keepdim=*/true);

  if (out)
    _lp = lp_crystal.rotate(_R());

  if (dout_din)
  {
    if (_g.is_dependent())
      _lp.d(_g) = A.rotate(_R());

    if (_R.is_dependent())
      _lp.d(_R) = lp_crystal.drotate(_R());
  }
}
} // namespace neml2
