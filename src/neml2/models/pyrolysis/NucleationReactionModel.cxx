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

#include "neml2/models/pyrolysis/NucleationReactionModel.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/log.h"

namespace neml2
{
register_NEML2_object(NucleationReactionModel);
OptionSet
NucleationReactionModel::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Define the nucleation reaction model, takes the form of \\f$ f = "
                  "k(1-a)(-ln(1-a))^n \\f$, where "
                  "\\f$ k \\f$ is the scaling constant, \\f$ n \\f$ is the reaction order, and "
                  "\\f$ a \\f$ is the reaction amount";

  options.set_parameter<TensorName<Scalar>>("scaling_constant");
  options.set("scaling_constant").doc() = "Scaling constant, k";

  options.set_parameter<TensorName<Scalar>>("reaction_order");
  options.set("reaction_order").doc() = "Reaction order, n";

  options.set_input("reaction_amount") = VariableName("state", "reaction_amount");
  options.set("reaction_amount").doc() = "Variable involes in the reaction, a";

  options.set_output("reaction_out") = VariableName("state", "out");
  options.set("reaction_out").doc() = "Chemical Reaction Model output, f.";

  return options;
}

NucleationReactionModel::NucleationReactionModel(const OptionSet & options)
  : Model(options),
    _k(declare_parameter<Scalar>("A", "scaling_constant")),
    _n(declare_parameter<Scalar>("Ea", "reaction_order")),
    _a(declare_input_variable<Scalar>("reaction_amount")),
    _f(declare_output_variable<Scalar>("reaction_out"))
{
}

void
NucleationReactionModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _f = _k * (1.0 - _a) * pow(-1.0 * log(1.0 - _a), _n);
  }

  if (dout_din)
  {
    _f.d(_a) = _k * _n * pow(-log(1 - _a), _n - 1) - _k * pow(-log(1 - _a), _n);
  }
}
}