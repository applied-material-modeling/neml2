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

#include "neml2/models/finite_volume/LinearReaction.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(LinearReaction);

OptionSet
LinearReaction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Linear reaction rate R = -lambda * u.";

  options.set_parameter<TensorName<Scalar>>("lambda");
  options.set("lambda").doc() = "Reaction parameter lambda.";

  options.set_input("u") = VariableName(STATE, "u");
  options.set("u").doc() = "Field values.";

  options.set_output("reaction") = VariableName(STATE, "R");
  options.set("reaction").doc() = "Reaction rate.";

  return options;
}

LinearReaction::LinearReaction(const OptionSet & options)
  : Model(options),
    _lambda(declare_parameter<Scalar>("lambda", "lambda", true)),
    _u(declare_input_variable<Scalar>("u")),
    _R(declare_output_variable<Scalar>("reaction"))
{
}

void
LinearReaction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _R = -_lambda * _u;

  if (dout_din)
  {
    const auto N = _u.intmd_size(-1);
    const auto I = intmd_diagonalize(imap_v<Scalar>(_u.options()).intmd_expand(N));

    // dR/du is a diagonal map over the intermediate dimension
    if (_u.is_dependent())
    {
      const auto lambda_vec = (_lambda.intmd_dim() == 0 ? _lambda.intmd_expand(N) : _lambda);
      _R.d(_u, 2, 1, 1) = (-lambda_vec).intmd_unsqueeze(1) * I;
    }

    if (const auto * const lambda = nl_param("lambda"))
    {
      if (lambda->intmd_dim() == 0)
        // lambda is a scalar: derivative is a vector over u's intermediate dimension
        _R.d(*lambda, 1, 1, 0) = -_u();
      else
        // lambda has matching intermediate dimension: diagonal map
        _R.d(*lambda, 2, 1, 1) = (-_u()).intmd_unsqueeze(1) * I;
    }
  }
}
} // namespace neml2
