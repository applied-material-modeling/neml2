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

#include <torch/script.h>

#include "neml2/models/Model.h"

namespace neml2
{
/**
 * @brief This class spits out the creep strain rate along with the rate of two other internal
 * variables, given the von Mises stress, temperature, and the current internal state as input.

 * Interestingly, the model is defined by a "neural network" loaded from a torch script. So this
 * example demonstrates the usage of pretrained machine learning model as part (or all) of the
 * material model.
 */
class TorchScriptFlowRate : public Model
{
public:
  TorchScriptFlowRate(const OptionSet & options);

  static OptionSet expected_options();

protected:
  void request_AD() override;

  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Model input
  // @{
  /// The von Mises stress
  const Variable<Scalar> & _s;
  /// Temperature
  const Variable<Scalar> & _T;
  /// Internal variables, could be grain size, stoichiometries, etc.
  const Variable<Scalar> & _G;
  const Variable<Scalar> & _C;
  // @}

  /// Model output
  // @{
  /// Creep strain rate
  Variable<Scalar> & _ep_dot;
  /// Rate of the 1st internal state
  Variable<Scalar> & _G_dot;
  /// Rate of the 2nd internal state
  Variable<Scalar> & _C_dot;
  // @}

  /// The torch script to be used as the forward operator
  std::unique_ptr<torch::jit::script::Module> _surrogate;
};
}
