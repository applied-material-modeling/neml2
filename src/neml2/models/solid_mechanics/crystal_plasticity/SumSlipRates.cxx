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

#include "neml2/models/solid_mechanics/crystal_plasticity/SumSlipRates.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/sum.h"
#include "neml2/tensors/functions/sign.h"

namespace neml2
{
register_NEML2_object(SumSlipRates);

OptionSet
SumSlipRates::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculates the sum of the absolute value of all the slip rates as \\f$ "
                  "\\sum_{i=1}^{n_{slip}} \\left| \\dot{\\gamma}_i \\right| \\f$.";

  options.set_input("slip_rates") = VariableName(STATE, "internal", "slip_rates");
  options.set("slip_rates").doc() = "The name of individual slip rates";

  options.set_output("sum_slip_rates") = VariableName(STATE, "internal", "sum_slip_rates");
  options.set("sum_slip_rates").doc() = "The output name for the scalar sum of the slip rates";

  options.set<Size>("dim") = -1;
  options.set("dim").doc() = "The intermediate dimension over which to sum the slip rates.";

  return options;
}

SumSlipRates::SumSlipRates(const OptionSet & options)
  : Model(options),
    _sg(declare_output_variable<Scalar>("sum_slip_rates")),
    _g(declare_input_variable<Scalar>("slip_rates")),
    _dim(options.get<Size>("dim"))
{
}

void
SumSlipRates::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _sg = intmd_sum(abs(_g()), _dim, /*keepdim=*/false);

  if (dout_din)
    if (_g.is_dependent())
    {
      auto I = TensorShape(_g.intmd_sizes());
      // I[utils::normalize_dim(_dim, 0, _g.intmd_dim())] = 1;
      I.erase(I.begin() + utils::normalize_dim(_dim, 0, _g.intmd_dim()));
      _sg.d(_g) = sign(_g()).intmd_expand(utils::add_shapes(I, _g.intmd_sizes()));
    }
}

} // namespace neml2
