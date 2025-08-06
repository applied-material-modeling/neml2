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

#include "neml2/models/ImplicitListReduction.h"
#include "neml2/tensors/functions/mean.h"

namespace neml2
{
template <typename T>
OptionSet
ImplicitListReduction<T>::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Reduce a list tensor to a single tensor by averaging over the list dimension. "
                  "The residual is defined as \\f$ r = \\hat{s} - \\frac{1}{n} \\sum_i^n s_i \\f$, "
                  "where \\f$s_i\\f$ are the "
                  "values in the list dimension, and \\f$\\hat{s}\\f$ is the reduced value.";

  NonlinearSystem::enable_automatic_scaling(options);

  options.set_input("variable");
  options.set("variable").doc() = "Averaged variable";

  options.set_input("batched_variable");
  options.set("batched_variable").doc() = "Batched collection of values";

  options.set<Size>("batched_variable_list_size");
  options.set("batched_variable_list_size").doc() =
      "Size of the list dimension in the batched variable.  This cannot be infered when the model "
      "is called and must remain fixed.";

  return options;
}

template <typename T>
ImplicitListReduction<T>::ImplicitListReduction(const OptionSet & options)
  : Model(options),
    _batched_list_size(options.get<Size>("batched_variable_list_size")),
    _val(declare_input_variable<T>("variable")),
    _batched_vals(declare_input_variable<T>("batched_variable", _batched_list_size)),
    _r(declare_output_variable<T>(_val.name().remount(RESIDUAL)))
{
}

template <typename T>
void
ImplicitListReduction<T>::diagnose() const
{
  Model::diagnose();
  diagnostic_assert_state(_val);
  diagnostic_assert_state(_batched_vals);
}

template <typename T>
void
ImplicitListReduction<T>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (_batched_vals.list_size(0) != _batched_list_size)
    throw NEMLException("Input batched tensor list dimension does not match user input.");

  if (out)
  {
    _r = _val - batch_mean(_batched_vals, -1);
  }

  if (dout_din)
  {
    _r.d(_val) = T::identity_map(_val.options());

    const auto I = T::identity_map(_batched_vals.options())
                       .base_reshape({T::const_base_storage, T::const_base_storage})
                       .batch_expand_as(T(_batched_vals));
    auto A =
        Tensor(-I / _batched_vals.list_size(0), _batched_vals.batch_sizes()).base_transpose(0, 1);
    _r.d(_batched_vals) = A;
  }
}

#define REGISTER(T)                                                                                \
  using T##ImplicitListReduction = ImplicitListReduction<T>;                                       \
  register_NEML2_object(T##ImplicitListReduction);                                                 \
  template class ImplicitListReduction<T>
REGISTER(Scalar);
REGISTER(Vec);
REGISTER(SR2);
} // namespace neml2
