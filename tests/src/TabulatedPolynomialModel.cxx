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

#include "TabulatedPolynomialModel.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(TabulatedPolynomialModel);

OptionSet
TabulatedPolynomialModel::expected_options()
{
  auto options = Model::expected_options();
  // Model inputs
  options.set<VariableName>("von_mises_stress") = VariableName(STATE, "s");
  options.set<VariableName>("temperature") = VariableName(FORCES, "T");
  options.set<VariableName>("internal_state_1") = VariableName(STATE, "s1");
  options.set<VariableName>("internal_state_2") = VariableName(STATE, "s2");
  // Model outputs
  options.set<VariableName>("equivalent_plastic_strain_rate") = VariableName(STATE, "ep_rate");
  options.set<VariableName>("internal_state_1_rate") = VariableName(STATE, "s1_rate");
  options.set<VariableName>("internal_state_2_rate") = VariableName(STATE, "s2_rate");
  // Model constants
  options.set<TensorName<Tensor>>("A0");
  options.set<TensorName<Tensor>>("A1");
  options.set<TensorName<Tensor>>("A2");
  options.set<TensorName<Tensor>>("stress_tile_lower_bounds");
  options.set<TensorName<Tensor>>("stress_tile_upper_bounds");
  options.set<TensorName<Tensor>>("temperature_tile_lower_bounds");
  options.set<TensorName<Tensor>>("temperature_tile_upper_bounds");
  options.set<double>("index_sharpness") = 1.0;
  return options;
}

TabulatedPolynomialModel::TabulatedPolynomialModel(const OptionSet & options)
  : Model(options),
    _A0(declare_buffer<Tensor>("A0", "A0")),
    _A1(declare_buffer<Tensor>("A1", "A1")),
    _A2(declare_buffer<Tensor>("A2", "A2")),
    _s_lb(declare_buffer<Tensor>("s_lb", "stress_tile_lower_bounds")),
    _s_ub(declare_buffer<Tensor>("s_ub", "stress_tile_upper_bounds")),
    _T_lb(declare_buffer<Tensor>("T_lb", "temperature_tile_lower_bounds")),
    _T_ub(declare_buffer<Tensor>("T_ub", "temperature_tile_upper_bounds")),
    _s(declare_input_variable<Scalar>("von_mises_stress")),
    _T(declare_input_variable<Scalar>("temperature")),
    _s1(declare_input_variable<Scalar>("internal_state_1")),
    _s2(declare_input_variable<Scalar>("internal_state_2")),
    _ep_dot(declare_output_variable<Scalar>("equivalent_plastic_strain_rate")),
    _s1_dot(declare_output_variable<Scalar>("internal_state_1_rate")),
    _s2_dot(declare_output_variable<Scalar>("internal_state_2_rate")),
    _k(options.get<double>("index_sharpness"))
{
}

ATensor
TabulatedPolynomialModel::smooth_index(const ATensor & x,
                                       const ATensor & lb,
                                       const ATensor & ub) const
{
  auto x0 = x.unsqueeze(-1);
  return 0.5 * (at::sigmoid(_k * (x0 - lb)) - at::sigmoid(_k * (x0 - ub)));
}

void
TabulatedPolynomialModel::request_AD()
{
  std::vector<const VariableBase *> inputs = {&_s, &_T, &_s1, &_s2};
  _ep_dot.request_AD(inputs);
  _s1_dot.request_AD(inputs);
  _s2_dot.request_AD(inputs);
}

void
TabulatedPolynomialModel::set_value(bool out, bool /*dout_din*/, bool /*d2out_din2*/)
{
  // Broadcast and expand batch shape
  std::vector<Tensor> inputs = {_s(), _T(), _s1(), _s2()};
  const auto dynamic_sizes = utils::broadcast_dynamic_sizes(inputs);
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = inputs[i].dynamic_expand(dynamic_sizes);

  // This example model has 4 input variables:
  //
  //   von Mises stress
  //   temperature
  //   internal state 1
  //   internal state 2
  //
  // First concatenate them together and unsqueeze to shape (...; 1, 1, 4, 1) so that we can do
  // batched matrix vector product
  ATensor x = base_stack(inputs, /*dim=*/-1);
  x = x.unsqueeze(-2).unsqueeze(-2).unsqueeze(-1);

  // The smooth index ij is of shape (...; 2, 3)
  auto i = smooth_index(_s(), _s_lb, _s_ub);
  auto j = smooth_index(_T(), _T_lb, _T_ub);
  auto ij = at::matmul(i.unsqueeze(-1), j.unsqueeze(-2));

  if (out)
  {
    // Map input to output (just a 2nd order polynomial)
    // After squeezing, y is of shape (...; 2, 3, 3)
    auto y = _A0.unsqueeze(-1) + at::matmul(_A1, x) + at::matmul(_A2, x * x);
    y = y.squeeze(-1);

    // Now y contains outputs from each cell of the table. We need to "select" the cell.
    // "Selecting" the cell is equivalent to contracting ij with y
    y = at::einsum("...ij,...ijk", {ij, y});
    _ep_dot = Scalar(y.index({indexing::Ellipsis, 0}), 0);
    _s1_dot = Scalar(y.index({indexing::Ellipsis, 1}), 0);
    _s2_dot = Scalar(y.index({indexing::Ellipsis, 2}), 0);
  }
}
}
