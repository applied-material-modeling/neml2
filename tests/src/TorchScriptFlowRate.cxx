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
#include "neml2/misc/math.h"

using namespace neml2;

register_NEML2_object(TorchScriptFlowRate);

OptionSet
TorchScriptFlowRate::expected_options()
{
  auto options = Model::expected_options();
  // Model inputs
  options.set<VariableName>("von_mises_stress") = VariableName("state", "s");
  options.set<VariableName>("temperature") = VariableName("forces", "T");
  options.set<VariableName>("internal_state_1") = VariableName("state", "G");
  options.set<VariableName>("internal_state_2") = VariableName("state", "C");
  // Model outputs
  options.set<VariableName>("equivalent_plastic_strain_rate") = VariableName("state", "ep_rate");
  options.set<VariableName>("internal_state_1_rate") = VariableName("state", "G_rate");
  options.set<VariableName>("internal_state_2_rate") = VariableName("state", "C_rate");
  // The machine learning model
  options.set<std::string>("torch_script");
  return options;
}

TorchScriptFlowRate::TorchScriptFlowRate(const OptionSet & options)
  : Model(options),
    _s(declare_input_variable<Scalar>("von_mises_stress")),
    _T(declare_input_variable<Scalar>("temperature")),
    _G(declare_input_variable<Scalar>("internal_state_1")),
    _C(declare_input_variable<Scalar>("internal_state_2")),
    _ep_dot(declare_output_variable<Scalar>("equivalent_plastic_strain_rate")),
    _G_dot(declare_output_variable<Scalar>("internal_state_1_rate")),
    _C_dot(declare_output_variable<Scalar>("internal_state_2_rate")),
    _surrogate(std::make_unique<torch::jit::script::Module>(
        torch::jit::load(options.get<std::string>("torch_script"))))
{
}

void
TorchScriptFlowRate::request_AD()
{
  std::vector<const VariableBase *> inputs = {&_s, &_T, &_G, &_C};
  _ep_dot.request_AD(inputs);
  _G_dot.request_AD(inputs);
  _C_dot.request_AD(inputs);
}

void
TorchScriptFlowRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!dout_din || !d2out_din2,
                  "Only AD derivatives are currently supported for this model");

  if (out)
  {
    // Broadcast and expand batch shape
    std::vector<Tensor> inputs = {_s, _T, _G, _C};
    const auto batch_sizes = utils::broadcast_batch_sizes(inputs);
    for (std::size_t i = 0; i < inputs.size(); ++i)
      inputs[i] = inputs[i].batch_expand(batch_sizes);

    // This example model has 4 input variables:
    //
    //   von Mises stress
    //   temperature
    //   internal state 1
    //   internal state 2
    //
    // First concatenate them together and unsqueeze to shape (...; 4)
    auto x = math::base_stack(inputs, /*dim=*/-1);

    // Send it through the surrogate model loaded from torch script
    auto y = _surrogate->forward({x}).toTensor();

    // Assuming the output is in the following order
    //
    //   equivalent plastic strain rate
    //   internal state 1 rate
    //   internal state 2 rate
    _ep_dot = Scalar(y.index({torch::indexing::Ellipsis, 0}));
    _G_dot = Scalar(y.index({torch::indexing::Ellipsis, 1}));
    _C_dot = Scalar(y.index({torch::indexing::Ellipsis, 2}));
  }
}
