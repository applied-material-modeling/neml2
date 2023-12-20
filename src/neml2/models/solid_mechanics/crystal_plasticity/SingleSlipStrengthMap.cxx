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

#include "neml2/models/solid_mechanics/crystal_plasticity/SingleSlipStrengthMap.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
register_NEML2_object(SingleSlipStrengthMap);

OptionSet
SingleSlipStrengthMap::expected_options()
{
  OptionSet options = SlipStrengthMap::expected_options();

  options.set<LabeledAxisAccessor>("slip_hardening") =
      vecstr{"state", "internal", "slip_hardening"};
  options.set<CrossRef<Scalar>>("constant_strength");

  return options;
}

SingleSlipStrengthMap::SingleSlipStrengthMap(const OptionSet & options)
  : SlipStrengthMap(options),
    slip_hardening(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("slip_hardening"))),
    _tau_const(declare_parameter<Scalar>("constant_strength", "constant_strength"))
{
  setup();
}

void
SingleSlipStrengthMap::set_value(const LabeledVector & in,
                                 LabeledVector * out,
                                 LabeledMatrix * dout_din,
                                 LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  // Grab the input
  const auto tau_bar = in.get<Scalar>(slip_hardening) + _tau_const;

  if (out)
    out->set_list(tau_bar.batch_unsqueeze(-1).batch_expand(utils::add_shapes(
                      tau_bar.batch_sizes(), TorchShape{crystal_geometry.nslip()})),
                  slip_strengths);

  if (dout_din)
    dout_din->set_list(BatchTensor::ones(utils::add_shapes(tau_bar.batch_sizes(),
                                                           TorchShape{crystal_geometry.nslip(), 1}),
                                         TorchShape{},
                                         tau_bar.dtype()),
                       slip_strengths,
                       slip_hardening);
}
} // namespace neml2
