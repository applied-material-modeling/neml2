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

#include "neml2/models/solid_mechanics/Normality.h"
#include "neml2/base/guards.h"

namespace neml2
{
register_NEML2_object(Normality);

OptionSet
Normality::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Store the first derivatives of a scalar-valued function in given variables, "
                  "i.e. \\f$ u_i = \\dfrac{f(\\boldsymbol{v})}{v_i} \\f$.";

  options.set<std::string>("model");
  options.set("model").doc() = "The model which evaluates the scalar-valued function";

  options.set<VariableName>("function");
  options.set("function").doc() = "Function to take derivative";

  options.set<std::vector<VariableName>>("from");
  options.set("from").doc() = "Function arguments to take derivatives w.r.t.";

  options.set<std::vector<VariableName>>("to");
  options.set("to").doc() = "Variables to store the first derivatives";

  return options;
}

Normality::Normality(const OptionSet & options)
  : Model(options),
    _model(register_model<Model>(options.get<std::string>("model"))),
    _f(options.get<VariableName>("function"))
{
  // Set up the conjugate pairs
  const auto from = options.get<std::vector<VariableName>>("from");
  const auto to = options.get<std::vector<VariableName>>("to");
  neml_assert(from.size() == to.size(),
              "The conjugate pairs should have a one-to-one correspondance. ",
              from.size(),
              " variables are being mapped to ",
              to.size(),
              " variables.");

  // Declare output variables
  for (size_t i = 0; i < from.size(); i++)
    _conjugate_pairs.emplace(from[i], clone_output_variable(_model.input_variable(from[i]), to[i]));
}

void
Normality::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Normality doesn't implement second derivatives.");

  {
    SolvingNonlinearSystem guard(false);
    if (out && !dout_din)
      _model.forward_maybe_jit(false, true, false);
    else
      _model.forward_maybe_jit(false, true, true);
  }

  const auto & fvar = _model.output_variable(_f);
  for (auto && [iname, ivar] : _conjugate_pairs)
  {
    if (!fvar.derivatives().count(iname))
    {
      (*ivar) = Tensor::zeros(ivar->base_sizes(), fvar.options());
      continue;
    }

    if (out)
      (*ivar) = fvar.derivatives().at(iname).base_reshape(ivar->base_sizes());

    if (dout_din)
      for (auto && [jname, jvar] : _model.input_variables())
        if (jvar.is_dependent() && fvar.second_derivatives().count(iname) &&
            fvar.second_derivatives().at(iname).count(jname))
          ivar->d(jvar) = fvar.second_derivatives().at(iname).at(jname);
  }
}
} // namespace neml2
