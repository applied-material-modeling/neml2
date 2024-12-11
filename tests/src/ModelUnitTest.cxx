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

#include "ModelUnitTest.h"
#include "utils.h"
#include "neml2/misc/math.h"

#include <torch/cuda.h>

namespace neml2
{
register_NEML2_object(ModelUnitTest);

OptionSet
ModelUnitTest::expected_options()
{
  OptionSet options = Driver::expected_options();
  options.set<std::string>("model");
  options.set<TensorShape>("batch_shape") = {1};
  options.set<bool>("check_values") = true;
  options.set<bool>("check_first_derivatives") = true;
  options.set<bool>("check_second_derivatives") = false;
  options.set<bool>("check_AD_first_derivatives") = true;
  options.set<bool>("check_AD_second_derivatives") = false;
  options.set<bool>("check_AD_derivatives") = false;
  options.set<bool>("check_AD_parameter_derivatives") = true;
  options.set<bool>("check_cuda") = true;
  options.set<bool>("check_disable_AD") = true;
  options.set<std::vector<VariableName>>("input_batch_tensor_names");
  options.set<std::vector<CrossRef<Tensor>>>("input_batch_tensor_values");
  options.set<std::vector<VariableName>>("output_batch_tensor_names");
  options.set<std::vector<CrossRef<Tensor>>>("output_batch_tensor_values");
  options.set<std::vector<VariableName>>("input_scalar_names");
  options.set<std::vector<CrossRef<Scalar>>>("input_scalar_values");
  options.set<std::vector<VariableName>>("input_symr2_names");
  options.set<std::vector<CrossRef<SR2>>>("input_symr2_values");
  options.set<std::vector<VariableName>>("input_skewr2_names");
  options.set<std::vector<CrossRef<WR2>>>("input_skewr2_values");
  options.set<std::vector<VariableName>>("output_scalar_names");
  options.set<std::vector<CrossRef<Scalar>>>("output_scalar_values");
  options.set<std::vector<VariableName>>("output_symr2_names");
  options.set<std::vector<CrossRef<SR2>>>("output_symr2_values");
  options.set<std::vector<VariableName>>("output_skewr2_names");
  options.set<std::vector<CrossRef<WR2>>>("output_skewr2_values");
  options.set<std::vector<VariableName>>("input_rot_names");
  options.set<std::vector<CrossRef<Rot>>>("input_rot_values");
  options.set<std::vector<VariableName>>("output_rot_names");
  options.set<std::vector<CrossRef<Rot>>>("output_rot_values");
  options.set<Real>("output_rel_tol") = 1e-5;
  options.set<Real>("output_abs_tol") = 1e-8;
  options.set<Real>("derivatives_rel_tol") = 1e-5;
  options.set<Real>("derivatives_abs_tol") = 1e-8;
  options.set<Real>("second_derivatives_rel_tol") = 1e-5;
  options.set<Real>("second_derivatives_abs_tol") = 1e-8;
  options.set<Real>("parameter_derivatives_rel_tol") = 1e-5;
  options.set<Real>("parameter_derivatives_abs_tol") = 1e-8;
  return options;
}

ModelUnitTest::ModelUnitTest(const OptionSet & options)
  : Driver(options),
    _model(get_model(options.get<std::string>("model"), true)),
    _model_disable_AD(get_model(options.get<std::string>("model"), false)),
    _batch_shape(options.get<TensorShape>("batch_shape")),
    _check_values(options.get<bool>("check_values")),
    _check_1st_deriv(options.get<bool>("check_first_derivatives")),
    _check_2nd_deriv(options.get<bool>("check_second_derivatives")),
    _check_AD_1st_deriv(options.get<bool>("check_AD_first_derivatives")),
    _check_AD_2nd_deriv(options.get<bool>("check_AD_second_derivatives")),
    _check_AD_derivs(options.get<bool>("check_AD_derivatives")),
    _check_AD_param_derivs(options.get<bool>("check_AD_parameter_derivatives")),
    _check_cuda(options.get<bool>("check_cuda")),
    _check_disable_AD(options.get<bool>("check_disable_AD")),
    _deriv_order(-1),
    _out_rtol(options.get<Real>("output_rel_tol")),
    _out_atol(options.get<Real>("output_abs_tol")),
    _deriv_rtol(options.get<Real>("derivatives_rel_tol")),
    _deriv_atol(options.get<Real>("derivatives_abs_tol")),
    _secderiv_rtol(options.get<Real>("second_derivatives_rel_tol")),
    _secderiv_atol(options.get<Real>("second_derivatives_abs_tol")),
    _param_rtol(options.get<Real>("parameter_derivatives_rel_tol")),
    _param_atol(options.get<Real>("parameter_derivatives_abs_tol"))
{
  _in = LabeledVector::zeros(_batch_shape, {&_model.input_axis()});
  fill_vector<Tensor>(_in, "input_batch_tensor_names", "input_batch_tensor_values");
  fill_vector<Scalar>(_in, "input_scalar_names", "input_scalar_values");
  fill_vector<SR2>(_in, "input_symr2_names", "input_symr2_values");
  fill_vector<WR2>(_in, "input_skewr2_names", "input_skewr2_values");
  fill_vector<Rot>(_in, "input_rot_names", "input_rot_values");

  _out = LabeledVector::zeros(_batch_shape, {&_model.output_axis()});
  fill_vector<Tensor>(_out, "output_batch_tensor_names", "output_batch_tensor_values");
  fill_vector<Scalar>(_out, "output_scalar_names", "output_scalar_values");
  fill_vector<SR2>(_out, "output_symr2_names", "output_symr2_values");
  fill_vector<WR2>(_out, "output_skewr2_names", "output_skewr2_values");
  fill_vector<Rot>(_out, "output_rot_names", "output_rot_values");

  if (_check_2nd_deriv || _check_AD_2nd_deriv || _check_AD_derivs)
    _deriv_order = 2;
  else if (_check_1st_deriv || _check_AD_1st_deriv)
    _deriv_order = 1;
  else
    _deriv_order = 0;
}

bool
ModelUnitTest::run()
{
  if (!run(_model))
    return false;

  if (_check_disable_AD)
    if (!run(_model_disable_AD))
      return false;

  return true;
}

bool
ModelUnitTest::run(Model & model)
{
  model.reinit(_in, _deriv_order);
  check_all(model);

  if (_check_cuda && torch::cuda::is_available())
  {
    _in = _in.to(torch::kCUDA);
    model.reinit(_in, _deriv_order);
    check_all(model);
  }

  return true;
}

void
ModelUnitTest::check_all(Model & model)
{
  if (_check_values)
    check_values(model);

  if (_check_1st_deriv)
    check_derivatives(model, false, false);

  if (_check_2nd_deriv)
    check_second_derivatives(model, false, false);

  // When AD is enabled
  if (model.is_AD_enabled())
  {
    if (_check_AD_1st_deriv)
      check_derivatives(model, true, true);

    if (_check_AD_2nd_deriv)
      check_second_derivatives(model, false, true);

    if (_check_AD_derivs)
      check_second_derivatives(model, true, true);

    if (_check_AD_param_derivs)
      check_AD_parameter_derivatives(model);
  }
}

void
ModelUnitTest::check_values(Model & model)
{
  auto out = model.value(_in);
  neml_assert(utils::allclose(out, _out.to(out.options()), _out_rtol, _out_atol),
              "The model gives values that are different from expected. The expected labels are:\n",
              _out.axis(0),
              "\nThe model gives the following labels:\n",
              out.axis(0),
              "\nThe expected values are:\n",
              _out.tensor(),
              "\nThe model gives the following values:\n",
              out.tensor());
}

void
ModelUnitTest::check_derivatives(Model & model, bool first, bool second)
{
  model.use_AD_derivatives(first, second);

  auto exact = std::get<1>(model.value_and_dvalue(_in));
  auto numerical = finite_differencing_derivative(
      [this, &model](const Tensor & x) { return model.value(LabeledVector(x, _in.axes())); }, _in);
  neml_assert(torch::allclose(exact, numerical, _deriv_rtol, _deriv_atol),
              "The model gives derivatives that are different from those given by finite "
              "differencing. The model gives:\n",
              exact.tensor(),
              "\nFinite differencing gives:\n",
              numerical);

  auto exact2 = model.dvalue(_in);
  neml_assert(
      torch::allclose(exact, exact2, _deriv_rtol, _deriv_atol),
      "Derivatives computed by value_and_dvalue and dvalue disagree. valud_and_dvalue yields:\n",
      exact.tensor(),
      "\ndvalue yields:\n",
      exact2.tensor());
}

void
ModelUnitTest::check_second_derivatives(Model & model, bool first, bool second)
{
  model.use_AD_derivatives(first, second);
  auto exact = std::get<2>(model.value_and_dvalue_and_d2value(_in));
  auto numerical = finite_differencing_derivative(
      [this, &model](const Tensor & x)
      { return std::get<1>(model.value_and_dvalue(LabeledVector(x, _in.axes()))); },
      _in);
  neml_assert(torch::allclose(exact, numerical, _secderiv_rtol, _secderiv_atol),
              "The model gives second derivatives that are different from those given by finite "
              "differencing. The model gives:\n",
              exact.tensor(),
              "\nFinite differencing gives:\n",
              numerical);

  auto exact2 = std::get<1>(model.dvalue_and_d2value(_in));
  neml_assert(torch::allclose(exact, exact2, _deriv_rtol, _deriv_atol),
              "Second derivatives computed by value_and_dvalue_and_d2value and dvalue_and_d2value "
              "disagree. value_and_dvalue_and_d2value yields:\n",
              exact.tensor(),
              "\nvalue_and_d2value yields:\n",
              exact2.tensor());

  auto exact3 = model.d2value(_in);
  neml_assert(torch::allclose(exact, exact3, _deriv_rtol, _deriv_atol),
              "Second derivatives computed by value_and_dvalue_and_d2value and d2value "
              "disagree. value_and_dvalue_and_d2value yields:\n",
              exact.tensor(),
              "\nd2value yields:\n",
              exact3.tensor());
}

void
ModelUnitTest::check_AD_parameter_derivatives(Model & model)
{
  // Turn on AD for parameters
  for (auto && [name, param] : model.named_parameters())
  {
    auto param_expanded = Tensor(param).batch_expand_copy(model.batch_sizes());
    param = param_expanded;
    param.requires_grad_(true);
  }

  // Evaluate the model
  auto out = model.value(_in);

  // Extract AD parameter derivatives
  std::map<std::string, Tensor> exact;
  for (auto && [name, param] : model.named_parameters())
    exact[name] = math::jacrev(out, param);

  // Compare results against FD
  for (auto && [name, param] : model.named_parameters())
  {
    auto numerical = finite_differencing_derivative(
        [&, &name = name](const Tensor & x)
        {
          auto p0 = Tensor(model.get_parameter(name)).clone();
          model.set_parameter(name, x);
          auto out = model.value(_in);
          model.set_parameter(name, p0);
          return out;
        },
        param);
    neml_assert(torch::allclose(exact[name], numerical, _param_rtol, _param_atol),
                "The model gives derivatives for parameter '",
                name,
                "' that are different from those given by finite "
                "differencing. The model gives:\n",
                exact[name],
                "\nFinite differencing gives:\n",
                numerical);
  }
}
}
