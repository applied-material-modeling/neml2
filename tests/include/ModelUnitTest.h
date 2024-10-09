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

#pragma once

#include "neml2/drivers/Driver.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
class ModelUnitTest : public Driver
{
public:
  static OptionSet expected_options();

  ModelUnitTest(const OptionSet & options);

  bool run() override;

private:
  template <typename T>
  void set_variable(std::map<VariableName, Tensor> & storage,
                    const std::string & option_vars,
                    const std::string & option_vals)
  {
    const auto vars = input_options().get<std::vector<VariableName>>(option_vars);
    const auto vals = input_options().get<std::vector<CrossRef<T>>>(option_vals);
    neml_assert(vars.size() == vals.size(),
                "Trying to assign ",
                vals.size(),
                " values to ",
                vars.size(),
                " variables.");
    for (size_t i = 0; i < vars.size(); i++)
      storage[vars[i]] = T(vals[i]);
  }

  void check_all();
  void check_value();
  void check_dvalue();
  void check_d2value();
  void check_AD_parameter_derivatives();

  Model & _model;
  const bool _check_values;
  const bool _check_derivs;
  const bool _check_secderivs;
  const bool _check_AD_param_derivs;
  const bool _check_cuda;

  std::map<VariableName, Tensor> _in;
  std::map<VariableName, Tensor> _out;

  Real _val_rtol;
  Real _val_atol;
  Real _deriv_rtol;
  Real _deriv_atol;
  Real _secderiv_rtol;
  Real _secderiv_atol;
  Real _param_rtol;
  Real _param_atol;
};
} // namespace neml2
