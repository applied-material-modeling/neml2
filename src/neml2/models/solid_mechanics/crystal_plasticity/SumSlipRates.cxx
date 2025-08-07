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

#include "neml2/models/crystallography/CrystalGeometry.h"

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

  options.set<std::string>("crystal_geometry_name") = "crystal_geometry";
  options.set("crystal_geometry_name").doc() =
      "The name of the Data object containing the crystallographic information";

  return options;
}

SumSlipRates::SumSlipRates(const OptionSet & options)
  : Model(options),
    _crystal_geometry(register_data<crystallography::CrystalGeometry>(
        options.get<std::string>("crystal_geometry_name"))),
    _sg(declare_output_variable<Scalar>("sum_slip_rates")),
    _g(declare_input_variable<Scalar>("slip_rates", _crystal_geometry.nslip()))
{
}

void
SumSlipRates::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _sg = batch_sum(abs(_g.value()), 0);

  if (dout_din)
    if (_g.is_dependent())
      _sg.d(_g) = Tensor(sign(_g.value()).batch_unsqueeze(-1), _g.batch_dim());
}

} // namespace neml2
