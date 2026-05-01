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

#include "neml2/models/common/ConstantExtrapolationPredictor.h"
#include "neml2/tensors/macros.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
register_NEML2_object(ConstantExtrapolationPredictor);

OptionSet
ConstantExtrapolationPredictor::expected_options()
{
  OptionSet options = Predictor::expected_options();
  options.doc() =
      "Use constant extrapolation as the initial guess for the unknowns at the current time step.";

#define addOption(T)                                                                               \
  options.add_optional<std::vector<VariableName>>("unknowns_" #T,                                  \
                                                  "The unknowns to extrapolate of type " #T)
  FOR_ALL_PRIMITIVETENSOR(addOption);
#undef addOption

  return options;
}

ConstantExtrapolationPredictor::ConstantExtrapolationPredictor(const OptionSet & options)
  : Predictor(options)
{
#define getVar(T)                                                                                  \
  if (options.defined("unknowns_" #T))                                                             \
  {                                                                                                \
    auto names = options.get<std::vector<VariableName>>("unknowns_" #T);                           \
    for (const auto & name : names)                                                                \
    {                                                                                              \
      _var_n_##T.push_back(&declare_input_variable<T>(history_name(name, 1)));                     \
      _var_##T.push_back(&declare_output_variable<T>(name));                                       \
    }                                                                                              \
  }                                                                                                \
  static_assert(true)
  FOR_ALL_PRIMITIVETENSOR(getVar);
#undef getVar
}

void
ConstantExtrapolationPredictor::predict()
{
#define predictVar(T)                                                                              \
  for (std::size_t i = 0; i < _var_n_##T.size(); i++)                                              \
    *_var_##T[i] = (*_var_n_##T[i])();                                                             \
  static_assert(true)
  FOR_ALL_PRIMITIVETENSOR(predictVar);
#undef predictVar
}
} // namespace neml2
