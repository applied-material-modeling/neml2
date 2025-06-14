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

#include "neml2/models/solid_mechanics/AssociativeJ2FlowDirection.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(AssociativeJ2FlowDirection);

OptionSet
AssociativeJ2FlowDirection::expected_options()
{
  auto options = Model::expected_options();
  options.doc() = "The plastic flow direction assuming an associative J2 flow.";

  options.set<VariableName>("mandel_stress") = VariableName(STATE, "M");
  options.set("mandel_stress").doc() = "Mandel stress";

  options.set<VariableName>("flow_direction") = VariableName(STATE, "NM");
  options.set("flow_direction").doc() = "Flow direction";

  return options;
}

AssociativeJ2FlowDirection::AssociativeJ2FlowDirection(const OptionSet & options)
  : Model(options),
    _M(declare_input_variable<SR2>("mandel_stress")),
    _N(declare_output_variable<SR2>("flow_direction"))
{
}

void
AssociativeJ2FlowDirection::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto eps = machine_precision(_M.scalar_type());
  auto S = SR2(_M).dev();
  auto vm = std::sqrt(3.0 / 2.0) * S.norm(eps);
  auto dvm_dM = 3.0 / 2.0 * S / vm;

  if (out)
  {
    _N = dvm_dM;
  }

  if (dout_din)
    if (_M.is_dependent())
    {
      auto I = SSR4::identity_sym(_M.options());
      auto J = SSR4::identity_dev(_M.options());

      _N.d(_M) = 3.0 / 2.0 * (I - 2.0 / 3.0 * dvm_dM.outer(dvm_dM)) * J / vm;
    }
}
} // namespace neml2
