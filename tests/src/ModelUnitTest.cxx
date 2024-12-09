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
  options.set<bool>("check_values") = true;
  options.set<bool>("check_derivatives") = true;
  options.set<bool>("check_second_derivatives") = false;
  options.set<bool>("check_AD_parameter_derivatives") = false;
  options.set<bool>("check_cuda") = false;

  options.set<Real>("value_rel_tol") = 1e-5;
  options.set<Real>("value_abs_tol") = 1e-8;
  options.set<Real>("derivative_rel_tol") = 1e-5;
  options.set<Real>("derivative_abs_tol") = 1e-8;
  options.set<Real>("second_derivative_rel_tol") = 1e-5;
  options.set<Real>("second_derivative_abs_tol") = 1e-8;
  options.set<Real>("parameter_derivative_rel_tol") = 1e-5;
  options.set<Real>("parameter_derivative_abs_tol") = 1e-8;

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
  return options;
}

ModelUnitTest::ModelUnitTest(const OptionSet & options)
  : Driver(options),
    _model(get_model(options.get<std::string>("model"))),
    _check_values(options.get<bool>("check_values")),
    _check_derivs(options.get<bool>("check_derivatives")),
    _check_secderivs(options.get<bool>("check_second_derivatives")),
    _check_AD_param_derivs(options.get<bool>("check_AD_parameter_derivatives")),
    _check_cuda(options.get<bool>("check_cuda")),

    _val_rtol(options.get<Real>("value_rel_tol")),
    _val_atol(options.get<Real>("value_abs_tol")),
    _deriv_rtol(options.get<Real>("derivative_rel_tol")),
    _deriv_atol(options.get<Real>("derivative_abs_tol")),
    _secderiv_rtol(options.get<Real>("second_derivative_rel_tol")),
    _secderiv_atol(options.get<Real>("second_derivative_abs_tol")),
    _param_rtol(options.get<Real>("parameter_derivative_rel_tol")),
    _param_atol(options.get<Real>("parameter_derivative_abs_tol"))
{
  set_variable<Tensor>(_in, "input_batch_tensor_names", "input_batch_tensor_values");
  set_variable<Scalar>(_in, "input_scalar_names", "input_scalar_values");
  set_variable<SR2>(_in, "input_symr2_names", "input_symr2_values");
  set_variable<WR2>(_in, "input_skewr2_names", "input_skewr2_values");
  set_variable<Rot>(_in, "input_rot_names", "input_rot_values");

  set_variable<Tensor>(_out, "output_batch_tensor_names", "output_batch_tensor_values");
  set_variable<Scalar>(_out, "output_scalar_names", "output_scalar_values");
  set_variable<SR2>(_out, "output_symr2_names", "output_symr2_values");
  set_variable<WR2>(_out, "output_skewr2_names", "output_skewr2_values");
  set_variable<Rot>(_out, "output_rot_names", "output_rot_values");
}

bool
ModelUnitTest::run()
{
  check_all();

  if (_check_cuda && torch::cuda::is_available())
  {
    _model.to(torch::kCUDA);
    for (auto && [name, tensor] : _in)
      _in[name] = tensor.to(torch::kCUDA);

    check_all();
  }

  return true;
}

void
ModelUnitTest::check_all()
{
  if (_check_values)
    check_value();

  if (_check_derivs)
    check_dvalue();

  if (_check_secderivs)
    check_d2value();

  if (_check_AD_param_derivs)
    check_AD_parameter_derivatives();
}

void
ModelUnitTest::check_value()
{
  const auto out = _model.value(_in);

  neml_assert(out.size() == _out.size(),
              "The model gives a different number of outputs than "
              "expected. The expected number of outputs is ",
              _out.size(),
              " but the model gives ",
              out.size(),
              " outputs.");

  for (const auto & [name, expected_value] : _out)
  {
    neml_assert(
        out.find(name) != out.end(), "The model is missing the expected output '", name, "'.");
    neml_assert(torch::allclose(out.at(name), expected_value, _val_rtol, _val_atol),
                "The model gives values that are different from expected for output '",
                name,
                "'. The expected values are:\n",
                expected_value,
                "\nThe model gives the following values:\n",
                out.at(name));
  }
}

void
ModelUnitTest::check_dvalue()
{
  const auto exact = _model.dvalue(_in);

  for (const auto & yname : _model.output_axis().variable_names())
    for (const auto & xname : _model.input_axis().variable_names())
    {
      const auto x0 = _in.count(xname) ? _in.at(xname).base_flatten()
                                       : Tensor::zeros(_model.input_axis().variable_size(xname),
                                                       _model.tensor_options());
      auto numerical = finite_differencing_derivative(
          [this, &yname, &xname](const Tensor & x)
          {
            auto in = _in;
            in[xname] = x;
            return _model.value(in)[yname].base_flatten();
          },
          x0);

      // If the derivative does not exist, the numerical derivative should be zero
      if (!exact.count(yname) || !exact.at(yname).count(xname))
        neml_assert(
            torch::allclose(numerical, torch::zeros_like(numerical), _deriv_rtol, _deriv_atol),
            "The model gives zero derivatives for the output '",
            yname,
            "' with respect to '",
            xname,
            "', but finite differencing gives:\n",
            numerical);
      // Otherwise, the numerical derivative should be close to the exact derivative
      else
        neml_assert(
            torch::allclose(exact.at(yname).at(xname), numerical, _deriv_rtol, _deriv_atol),
            "The model gives derivatives that are different from finite differencing for output '",
            yname,
            "' with respect to '",
            xname,
            "'. The model gives:\n",
            exact.at(yname).at(xname),
            "\nFinite differencing gives:\n",
            numerical);
    }
}

void
ModelUnitTest::check_d2value()
{
  const auto exact = _model.d2value(_in);

  for (const auto & yname : _model.output_axis().variable_names())
    for (const auto & x1name : _model.input_axis().variable_names())
      for (const auto & x2name : _model.input_axis().variable_names())
      {
        const auto x20 =
            _in.count(x2name)
                ? _in.at(x2name).base_flatten()
                : Tensor::zeros(_model.input_axis().variable_size(x2name), _model.tensor_options());
        auto numerical = finite_differencing_derivative(
            [this, &yname, &x1name, &x2name](const Tensor & x)
            {
              auto in = _in;
              in[x2name] = x;
              return _model.dvalue(in)[yname][x1name];
            },
            x20);

        // If the derivative does not exist, the numerical derivative should be zero
        if (!exact.count(yname) || !exact.at(yname).count(x1name) ||
            !exact.at(yname).at(x1name).count(x2name))
          neml_assert(torch::allclose(
                          numerical, torch::zeros_like(numerical), _secderiv_rtol, _secderiv_atol),
                      "The model gives zero second derivatives for the output '",
                      yname,
                      "' with respect to '",
                      x1name,
                      "' and '",
                      x2name,
                      "', but finite differencing gives:\n",
                      numerical);
        // Otherwise, the numerical derivative should be close to the exact derivative
        else
          neml_assert(
              torch::allclose(
                  exact.at(yname).at(x1name).at(x2name), numerical, _secderiv_rtol, _secderiv_atol),
              "The model gives second derivatives that are different from finite "
              "differencing for output "
              "'",
              yname,
              "' with respect to '",
              x1name,
              "' and '",
              x2name,
              "'. The model gives:\n",
              exact.at(yname).at(x1name).at(x2name),
              "\nFinite differencing gives:\n",
              numerical);
      }
}

void
ModelUnitTest::check_AD_parameter_derivatives()
{
  // Turn on AD for parameters
  for (auto && [name, param] : _model.named_parameters())
  {
    // auto param_expanded = Tensor(param).batch_expand_copy(model.batch_sizes());
    // param = param_expanded;
    param.requires_grad_(true);
  }

  // Evaluate the model
  auto out = _model.value(_in);

  // Extract AD parameter derivatives
  // std::map<std::string, Tensor> exact;
  // for (auto && [name, param] : model.named_parameters())
  //   exact[name] = math::jacrev(out, param);

  // Compare results against FD
  // for (auto && [name, param] : model.named_parameters())
  // {
  //   auto numerical = finite_differencing_derivative(
  //       [&, &name = name](const Tensor & x)
  //       {
  //         auto p0 = Tensor(model.get_parameter(name)).clone();
  //         model.set_parameter(name, x);
  //         auto out = model.value(_in);
  //         model.set_parameter(name, p0);
  //         return out;
  //       },
  //       param);
  //   neml_assert(torch::allclose(exact[name], numerical, _param_rtol, _param_atol),
  //               "The model gives derivatives for parameter '",
  //               name,
  //               "' that are different from those given by finite "
  //               "differencing. The model gives:\n",
  //               exact[name],
  //               "\nFinite differencing gives:\n",
  //               numerical);
  // }
}
}
