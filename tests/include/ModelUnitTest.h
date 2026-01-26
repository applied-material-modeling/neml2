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
  void check_all();
  void check_value();
  void check_dvalue();
  void check_d2value();
  void check_AD_parameter_derivatives();

  const std::shared_ptr<Model> _model;
  const bool _check_values;
  const bool _check_derivs;
  const bool _check_secderivs;
  const bool _check_AD_param_derivs;

  ValueMap _in;
  ValueMap _out;

  double _val_rtol;
  double _val_atol;
  double _deriv_rtol;
  double _deriv_atol;
  double _secderiv_rtol;
  double _secderiv_atol;
  double _param_rtol;
  double _param_atol;

  const bool _show_params;
};
} // namespace neml2
