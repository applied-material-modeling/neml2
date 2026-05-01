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

#include "neml2/models/common/LinearExtrapolationPredictor.h"
#include "neml2/tensors/macros.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/functions/abs.h"

namespace neml2
{
register_NEML2_object(LinearExtrapolationPredictor);

OptionSet
LinearExtrapolationPredictor::expected_options()
{
  OptionSet options = ConstantExtrapolationPredictor::expected_options();
  options.doc() =
      "Use temporal extrapolation assuming constant rate of change as the initial guess for the "
      "unknowns at the current time step. The linear extrapolation can be written as \\f$u = u_n + "
      "(u_n - u_{n-1}) \\frac{t - t_n}{t_n - t_{n-1}}\\f$, where \\f$u\\f$ is the unknown and "
      "\\f$n\\f$ is the time step counter, respectively.";

  options.add_input("time", "t", "Time");

  return options;
}

LinearExtrapolationPredictor::LinearExtrapolationPredictor(const OptionSet & options)
  : ConstantExtrapolationPredictor(options),
    _t(declare_input_variable<Scalar>("time")),
    _t_n(declare_input_variable<Scalar>(history_name(_t.name(), 1))),
    _t_nm1(declare_input_variable<Scalar>(history_name(_t.name(), 2)))
{
#define getVar(T)                                                                                  \
  if (options.defined("unknowns_" #T))                                                             \
  {                                                                                                \
    auto names = options.get<std::vector<VariableName>>("unknowns_" #T);                           \
    for (const auto & name : names)                                                                \
      _var_nm1_##T.push_back(&declare_input_variable<T>(history_name(name, 2)));                   \
  }                                                                                                \
  static_assert(true)
  FOR_ALL_PRIMITIVETENSOR(getVar);
#undef getVar
}

void
LinearExtrapolationPredictor::predict()
{
#define predictVar(T)                                                                              \
  for (std::size_t i = 0; i < _var_n_##T.size(); i++)                                              \
  {                                                                                                \
    auto v_extrap =                                                                                \
        *_var_n_##T[i] + (*_var_n_##T[i] - *_var_nm1_##T[i]) * (_t - _t_n) / (_t_n - _t_nm1);      \
    *_var_##T[i] = neml2::where(neml2::abs(_t_n - _t_nm1).batch_expand_as(v_extrap) >              \
                                    machine_precision(v_extrap.scalar_type()),                     \
                                v_extrap,                                                          \
                                (*_var_n_##T[i])());                                               \
  }                                                                                                \
  static_assert(true)
  FOR_ALL_PRIMITIVETENSOR(predictVar);
#undef predictVar
}
} // namespace neml2
