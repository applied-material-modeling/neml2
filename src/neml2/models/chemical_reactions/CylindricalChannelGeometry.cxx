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

#include "neml2/models/chemical_reactions/CylindricalChannelGeometry.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(CylindricalChannelGeometry);

OptionSet
CylindricalChannelGeometry::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the dimensionless inner and outer radii of the reaction product";

  options.set_input("solid_fraction") = VariableName{"state", "phi_s"};
  options.set("solid_fraction").doc() = "Volume fraction of the solid phase";
  options.set_input("product_fraction") = VariableName{"state", "phi_p"};
  options.set("product_fraction").doc() = "Volume fraction of the product phase";

  options.set_output("inner_radius") = VariableName{"state", "ri"};
  options.set("inner_radius").doc() = "Dimensionless inner radius of the product phase";
  options.set_output("outer_radius") = VariableName{"state", "ro"};
  options.set("outer_radius").doc() = "Dimensionless outer radius of the product phase";

  return options;
}

CylindricalChannelGeometry::CylindricalChannelGeometry(const OptionSet & options)
  : Model(options),
    _phi_s(declare_input_variable<Scalar>("solid_fraction")),
    _phi_p(declare_input_variable<Scalar>("product_fraction")),
    _ri(declare_output_variable<Scalar>("inner_radius")),
    _ro(declare_output_variable<Scalar>("outer_radius"))
{
}

void
CylindricalChannelGeometry::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto eps = machine_precision(_phi_s.scalar_type());
  const auto cap = 1 - _phi_s - _phi_p;
  const auto ri = sqrt(clamp(cap, eps, 1.0 - eps));
  const auto ro = sqrt(1 - _phi_s);

  if (out)
  {
    _ri = ri;
    _ro = ro;
  }

  if (dout_din)
  {
    _ri.d(_phi_s) = where(cap <= eps, Scalar::zeros_like(ri), -0.5 / ri);
    _ri.d(_phi_p) = where(cap <= eps, Scalar::zeros_like(ri), -0.5 / ri);
    _ro.d(_phi_s) = -0.5 / ro;
  }
}
} // namespace neml2
