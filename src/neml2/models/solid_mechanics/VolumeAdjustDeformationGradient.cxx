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

#include "neml2/models/solid_mechanics/VolumeAdjustDeformationGradient.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(VolumeAdjustDeformationGradient);

OptionSet
VolumeAdjustDeformationGradient::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the volume-adjusted deformation gradient, i.e. \\f$ F_e = "
                  "J^{-\\frac{1}/{3}} F \\f$, where \\f$ F \\f$ is the pre-adjusted deformation "
                  "gradient and \\f$ J \\f$ is the total jacobian of the volumetric deformation "
                  "gradients to be removed.";

  options.set_input("input");
  options.set("input").doc() = "Input deformation gradient";

  options.set_input("jacobian");
  options.set("jacobian").doc() =
      "The jacobian that controls the volume adjustment of the input deformation gradient";

  options.set_output("output");
  options.set("output").doc() = "Output adjusted deformation gradient";

  return options;
}

VolumeAdjustDeformationGradient::VolumeAdjustDeformationGradient(const OptionSet & options)
  : Model(options),
    _F(declare_input_variable<R2>("input")),
    _J(declare_input_variable<Scalar>("jacobian")),
    _Fe(declare_output_variable<R2>("output"))
{
}

void
VolumeAdjustDeformationGradient::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _Fe = _F * pow(_J, -1.0 / 3.0);

  if (dout_din)
  {
    if (_J.is_dependent())
      _Fe.d(_J) = _F * -1.0 / 3.0 * pow(_J, -4.0 / 3.0);

    if (_F.is_dependent())
    {
      auto I = R2::identity_map(_F.options());
      _Fe.d(_F) = pow(_J, -1.0 / 3.0) * I;
    }
  }
}
} // namespace neml2
