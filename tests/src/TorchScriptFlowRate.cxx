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

#include "TorchScriptFlowRate.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"

using namespace neml2;

register_NEML2_object(TorchScriptFlowRate);

OptionSet
TorchScriptFlowRate::expected_options()
{
  auto options = Model::expected_options();
  // Model inputs
  options.set<VariableName>("von_mises_stress") = VariableName(STATE, "s");
  options.set<VariableName>("temperature") = VariableName(FORCES, "T");
  // Model outputs
  options.set<VariableName>("equivalent_plastic_strain_rate") = VariableName(STATE, "ep_rate");
  // The machine learning model
  options.set<std::string>("torch_script");
  // No jitting :/
  options.set<bool>("jit") = false;
  options.set("jit").suppressed() = true;
  return options;
}

TorchScriptFlowRate::TorchScriptFlowRate(const OptionSet & options)
  : Model(options),
    _s(declare_input_variable<Scalar>("von_mises_stress")),
    _T(declare_input_variable<Scalar>("temperature")),
    _ep_dot(declare_output_variable<Scalar>("equivalent_plastic_strain_rate")),
    _surrogate(
        std::make_unique<jit::script::Module>(jit::load(options.get<std::string>("torch_script"))))
{
}

void
TorchScriptFlowRate::request_AD()
{
  std::vector<const VariableBase *> inputs = {&_s, &_T};
  _ep_dot.request_AD(inputs);
}

void
TorchScriptFlowRate::set_value(bool out, bool /*dout_din*/, bool /*d2out_din2*/)
{
  if (out)
  {
    // This example model has 4 input variables:
    //
    //   von Mises stress
    //   temperature
    //   internal state 1
    //   internal state 2
    //
    const auto G = Scalar::full(0.1, _s.options());
    const auto C = Scalar::full(0.2, _s.options());
    const jit::Stack x = {_s.value(), _T.value(), G, C};

    // Send it through the surrogate model loaded from torch script
    const auto y = _surrogate->forward(x).toTensor();

    // Eequivalent plastic strain rate
    _ep_dot = Scalar(y);
  }
}
