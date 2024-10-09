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

#include "neml2/models/LinearCombination.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ScalarLinearCombination);
register_NEML2_object(VecLinearCombination);
register_NEML2_object(SR2LinearCombination);

template <typename T>
OptionSet
LinearCombination<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = Model::expected_options();
  options.doc() = "Calculate linear combination of multiple " + tensor_type +
                  " tensors as \\f$ u = c_i v_i \\f$ (Einstein summation assumed), where \\f$ c_i "
                  "\\f$ are the coefficients, and \\f$ v_i \\f$ are the variables to be summed.";

  options.set<std::vector<VariableName>>("from_var");
  options.set("from_var").doc() = tensor_type + " tensors to be summed";

  options.set_output("to_var");
  options.set("to_var").doc() = "The sum";

  options.set_parameter<std::vector<CrossRef<Scalar>>>("coefficients") = {CrossRef<Scalar>("1")};
  options.set("coefficients").doc() = "Weights associated with each variable";

  options.set<bool>("coefficients_as_parameters") = false;
  options.set("coefficients_as_parameters").doc() =
      "By default, the coefficients are declared as buffers. Set this option to true to declare "
      "them as (trainable) parameters.";

  return options;
}

template <typename T>
LinearCombination<T>::LinearCombination(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<T>("to_var")),
    _coef_as_param(options.get<bool>("coefficients_as_parameters")),
    _coef(_coef_as_param ? declare_parameter<Tensor>("c", make_coef(options))
                         : declare_buffer<Tensor>("c", make_coef(options)))
{
  for (auto fv : options.get<std::vector<VariableName>>("from_var"))
    _from.push_back(&declare_input_variable<T>(fv));
}

template <typename T>
Tensor
LinearCombination<T>::make_coef(const OptionSet & options) const
{
  const auto coefs_in = options.get<std::vector<CrossRef<Scalar>>>("coefficients");
  const std::vector<Scalar> coefs(coefs_in.begin(), coefs_in.end());
  const std::vector<Tensor> coefs_tensors(coefs.begin(), coefs.end());
  return math::base_stack(coefs_tensors);
}

template <typename T>
void
LinearCombination<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const Size N = _from.size();

  if (out)
  {
    std::vector<Tensor> vals;
    for (auto from_var : _from)
      vals.push_back(from_var->value());

    // Broadcast and expand batch shape
    const auto batch_sizes = utils::broadcast_batch_sizes(vals);
    for (std::size_t i = 0; i < vals.size(); ++i)
      vals[i] = vals[i].batch_expand(batch_sizes);

    _to = math::batch_sum(Scalar(_coef, 1) * math::batch_stack(vals, -1), -1);
  }

  if (dout_din)
  {
    const auto deriv = Scalar(_coef, 1) * T::identity_map(_coef.options()).batch_expand(N);
    for (Size i = 0; i < N; i++)
      if (_from[i]->is_dependent())
        _to.d(*_from[i]) = deriv.batch_index({indexing::Ellipsis, i});
  }

  if (d2out_din2)
  {
    // zero
  }
}

template class LinearCombination<Scalar>;
template class LinearCombination<Vec>;
template class LinearCombination<SR2>;
} // namespace neml2
