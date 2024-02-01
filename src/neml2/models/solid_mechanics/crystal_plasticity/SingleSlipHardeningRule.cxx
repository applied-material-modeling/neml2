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

#include "neml2/models/solid_mechanics/crystal_plasticity/SingleSlipHardeningRule.h"

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/list_tensors.h"

using vecstr = std::vector<std::string>;

namespace neml2
{
OptionSet
SingleSlipHardeningRule::expected_options()
{
  OptionSet options = Model::expected_options();

  options.set<LabeledAxisAccessor>("slip_hardening_rate") =
      vecstr{"state", "internal", "slip_hardening_rate"};
  options.set<LabeledAxisAccessor>("slip_hardening") =
      vecstr{"state", "internal", "slip_hardening"};
  options.set<LabeledAxisAccessor>("sum_slip_rates") =
      vecstr{"state", "internal", "sum_slip_rates"};

  return options;
}

SingleSlipHardeningRule::SingleSlipHardeningRule(const OptionSet & options)
  : Model(options),
    slip_hardening_rate(
        declare_output_variable<Scalar>(options.get<LabeledAxisAccessor>("slip_hardening_rate"))),
    slip_hardening(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("slip_hardening"))),
    sum_slip_rates(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("sum_slip_rates")))
{
  setup();
}

} // namespace neml2