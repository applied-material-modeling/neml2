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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
template <typename T>
OptionSet
LinearCombination<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate linear combination of multiple " + tensor_type +
      " tensors as \\f$ u = c_i v_i + s \\f$ (Einstein summation assumed), where \\f$ c_i "
      "\\f$ are the coefficients, and \\f$ v_i \\f$ are the variables to be summed. \\f$ s \\f$ is "
      "a constant offset.";

  options.set<bool>("define_second_derivatives") = true;

  options.set<std::vector<VariableName>>("from_var");
  options.set("from_var").doc() = tensor_type + " tensors to be summed";

  options.set_output("to_var");
  options.set("to_var").doc() = "The sum";

  options.set_parameter<std::vector<TensorName<Scalar>>>("coefficients") = {
      TensorName<Scalar>("1")};
  options.set("coefficients").doc() =
      "Weights associated with each variable. This option takes a list of weights, one for each "
      "coefficient. When the length of this list is 1, the same weight applies to all "
      "coefficients.";

  options.set_parameter<TensorName<Scalar>>("constant_coefficient") = {TensorName<Scalar>("0")};
  options.set("constant_coefficient").doc() =
      "The constant coefficient added to the final summation";

  options.set<bool>("constant_coefficient_as_parameter") = false;
  options.set("constant_coefficient_as_parameter").doc() =
      "By default, the constant_coefficient are declared as buffers. Set this option to true to "
      "declare "
      "them as (trainable) parameters.";

  options.set<std::vector<bool>>("coefficient_as_parameter") = {false};
  options.set("coefficient_as_parameter").doc() =
      "By default, the coefficients are declared as buffers. Set this option to true to declare "
      "them as (trainable) parameters. This option takes a list of booleans, one for each "
      "coefficient. When the length of this list is 1, the boolean applies to all coefficients.";

  return options;
}

template <typename T>
LinearCombination<T>::LinearCombination(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<T>("to_var"))
{
  for (const auto & fv : options.get<std::vector<VariableName>>("from_var"))
    _from.push_back(&declare_input_variable<T>(fv));

  auto coef_as_param = options.get<std::vector<bool>>("coefficient_as_parameter");
  neml_assert(coef_as_param.size() == 1 || coef_as_param.size() == _from.size(),
              "Expected 1 or ",
              _from.size(),
              " entries in coefficient_as_parameter, got ",
              coef_as_param.size(),
              ".");

  // Expand the list of booleans to match the number of coefficients
  if (coef_as_param.size() == 1)
    coef_as_param = std::vector<bool>(_from.size(), coef_as_param[0]);

  const auto coef_refs = options.get<std::vector<TensorName<Scalar>>>("coefficients");
  neml_assert(coef_refs.size() == 1 || coef_refs.size() == _from.size(),
              "Expected 1 or ",
              _from.size(),
              " coefficients, got ",
              coef_refs.size(),
              ".");

  // Declare parameters or buffers
  _coefs.resize(_from.size());
  for (std::size_t i = 0; i < _from.size(); i++)
  {
    const auto & coef_ref = coef_refs.size() == 1 ? coef_refs[0] : coef_refs[i];
    if (coef_as_param[i])
      _coefs[i] =
          &declare_parameter<Scalar>("c_" + std::to_string(i), coef_ref, /*allow_nonlinear=*/true);
    else
      _coefs[i] = &declare_buffer<Scalar>("c_" + std::to_string(i), coef_ref);
  }

  if (options.user_specified("constant_coefficient"))
  {
    auto s_as_param = options.get<bool>("constant_coefficient_as_parameter");
    if (s_as_param)
      _s = &declare_parameter<Scalar>("s", "constant_coefficient", /*allow_nonlinear=*/true);
    else
      _s = &declare_buffer<Scalar>("s", "constant_coefficient");
  }
}

template <typename T>
void
LinearCombination<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
  {
    auto value = _s ? (*_s) + (*_coefs[0]) * (*_from[0]) : (*_coefs[0]) * (*_from[0]);
    for (std::size_t i = 1; i < _from.size(); i++)
      value = value + (*_coefs[i]) * (*_from[i]);
    _to = value;
  }

  if (dout_din)
  {
    const auto I = imap_v<T>(_from[0]->options());
    for (std::size_t i = 0; i < _from.size(); i++)
    {
      if (_from[i]->is_dependent())
        _to.d(*_from[i]) += (*_coefs[i]) * I;

      if (const auto * const pi = nl_param("c_" + std::to_string(i)))
        _to.d(*pi) += (*_from[i])();
    }
    if (const auto * const s = nl_param("s"))
      _to.d(*s) += neml2::Scalar::full(1.0, _from[0]->options());
  }

  if (d2out_din2)
  {
    // zero
  }
}

#define REGISTER(T)                                                                                \
  using T##LinearCombination = LinearCombination<T>;                                               \
  register_NEML2_object(T##LinearCombination);                                                     \
  template class LinearCombination<T>
REGISTER(Scalar);
REGISTER(Vec);
REGISTER(SR2);
REGISTER(R2);
} // namespace neml2
