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
#include "neml2/misc/types.h"
#include "neml2/tensors/functions/intrsc_intmd_dim_utils.h"
#include "neml2/tensors/functions/jacrev.h"
#include "neml2/misc/assertions.h"

#include "neml2/tensors/shape_utils.h"
#include "utils.h"

namespace neml2
{
template <typename T>
static void
set_variable(ValueMap & storage,
             const OptionSet & options,
             const std::string & option_vars,
             const std::string & option_vals)
{
  const auto & vars = options.get<std::vector<VariableName>>(option_vars);
  const auto & vals = options.get<std::vector<TensorName<T>>>(option_vals);
  neml_assert(vars.size() == vals.size(),
              "Trying to assign ",
              vals.size(),
              " values to ",
              vars.size(),
              " variables.");
  auto * factory = options.get<Factory *>("_factory");
  neml_assert(factory, "Failed assertion: factory != nullptr");
  for (size_t i = 0; i < vars.size(); i++)
    storage[vars[i]] = vals[i].resolve(factory);
}

register_NEML2_object(ModelUnitTest);

OptionSet
ModelUnitTest::expected_options()
{
  OptionSet options = Driver::expected_options();

  options.set<std::string>("model");
  options.set<bool>("check_values") = true;
  options.set<bool>("check_derivatives") = true;
  options.set<bool>("check_second_derivatives") = false;
  options.set<bool>("check_AD_parameter_derivatives") = true;

  options.set<double>("value_rel_tol") = 1e-5;
  options.set<double>("value_abs_tol") = 1e-8;
  options.set<double>("derivative_rel_tol") = 1e-5;
  options.set<double>("derivative_abs_tol") = 1e-8;
  options.set<double>("second_derivative_rel_tol") = 1e-5;
  options.set<double>("second_derivative_abs_tol") = 1e-8;
  options.set<double>("parameter_derivative_rel_tol") = 1e-5;
  options.set<double>("parameter_derivative_abs_tol") = 1e-8;

#define OPTION_SET_(T)                                                                             \
  options.set<std::vector<VariableName>>("input_" #T "_names");                                    \
  options.set<std::vector<TensorName<T>>>("input_" #T "_values");                                  \
  options.set<std::vector<VariableName>>("output_" #T "_names");                                   \
  options.set<std::vector<TensorName<T>>>("output_" #T "_values")
  FOR_ALL_TENSORBASE(OPTION_SET_);

  options.set<std::vector<VariableName>>("input_with_intrsc_intmd_dims");
  options.set<std::vector<Size>>("input_intrsc_intmd_dims");
  options.set<std::vector<VariableName>>("output_with_intrsc_intmd_dims");
  options.set<std::vector<Size>>("output_intrsc_intmd_dims");

  return options;
}

ModelUnitTest::ModelUnitTest(const OptionSet & options)
  : Driver(options),
    _model(factory()->get_model(options.get<std::string>("model"))),
    _check_values(options.get<bool>("check_values")),
    _check_derivs(options.get<bool>("check_derivatives")),
    _check_secderivs(options.get<bool>("check_second_derivatives")),
    _check_AD_param_derivs(options.get<bool>("check_AD_parameter_derivatives")),

    _val_rtol(options.get<double>("value_rel_tol")),
    _val_atol(options.get<double>("value_abs_tol")),
    _deriv_rtol(options.get<double>("derivative_rel_tol")),
    _deriv_atol(options.get<double>("derivative_abs_tol")),
    _secderiv_rtol(options.get<double>("second_derivative_rel_tol")),
    _secderiv_atol(options.get<double>("second_derivative_abs_tol")),
    _param_rtol(options.get<double>("parameter_derivative_rel_tol")),
    _param_atol(options.get<double>("parameter_derivative_abs_tol")),

    _input_intrsc_intmd_dims(options.get_map<VariableName, Size>("input_with_intrsc_intmd_dims",
                                                                 "input_intrsc_intmd_dims")),
    _output_intrsc_intmd_dims(options.get_map<VariableName, Size>("output_with_intrsc_intmd_dims",
                                                                  "output_intrsc_intmd_dims"))
{
#define SET_VARIABLE_(T)                                                                           \
  set_variable<T>(_in, options, "input_" #T "_names", "input_" #T "_values");                      \
  set_variable<T>(_out, options, "output_" #T "_names", "output_" #T "_values")
  FOR_ALL_TENSORBASE(SET_VARIABLE_);
}

bool
ModelUnitTest::run()
{
  check_all();

  for (const auto & device : get_test_suite_additional_devices())
  {
    _model->to(device);
    for (auto && [name, tensor] : _in)
      _in[name] = tensor.to(device);

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
  const auto out = _model->value(_in);

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
    neml_assert(at::allclose(out.at(name).to(kCPU), expected_value, _val_rtol, _val_atol),
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
  _model->clear_input();
  _model->assign_input(_in);
  _model->zero_undefined_input();
  _model->forward_maybe_jit(false, true, false);

  DerivMap exact;
  for (const auto & [yname, yvar] : _model->output_variables())
    for (const auto & [dy_dx, xvar] : yvar->derivatives())
      if (dy_dx.defined())
        exact[yname][xvar->name()] = pop_intrsc_intmd_dim(dy_dx);

  for (const auto & [yname, yvar] : _model->output_variables())
    for (auto & [xname, xvar] : _model->input_variables())
    {
      _model->clear_input();
      _model->assign_input(_in);
      _model->zero_undefined_input();

      const auto & x0 = xvar->tensor();
      const auto y_intrsc_intmd_dim = _output_intrsc_intmd_dims[yname];
      const auto x_intrsc_intmd_dim = _input_intrsc_intmd_dims[xname];
      const auto numerical = finite_differencing_derivative(
          [this, &xvar = xvar, &yvar = yvar, x_intrsc_intmd_dim, y_intrsc_intmd_dim](
              const Tensor & x)
          {
            (*xvar) = push_intrsc_intmd_dim(x, x_intrsc_intmd_dim);
            _model->forward_maybe_jit(true, false, false);
            return pop_intrsc_intmd_dim(yvar->tensor(), y_intrsc_intmd_dim);
          },
          pop_intrsc_intmd_dim(x0, x_intrsc_intmd_dim),
          1e-6,
          1e-6);

      // If the derivative does not exist, the numerical derivative should be zero
      if (!exact.count(yname) || !exact.at(yname).count(xname))
        neml_assert(at::allclose(numerical, at::zeros_like(numerical), _deriv_rtol, _deriv_atol),
                    "The model gives zero derivatives for the output '",
                    yname,
                    "' with respect to '",
                    xname,
                    "', but finite differencing gives:\n",
                    numerical);
      // Otherwise, the numerical derivative should be close to the exact derivative
      else
      {
        const auto & exact_val = exact.at(yname).at(xname);
        neml_assert(utils::sizes_broadcastable(exact_val.sizes(), numerical.sizes()),
                    "The model's and finite differencing derivatives for output '",
                    yname,
                    "' with respect to '",
                    xname,
                    "' have incompatible shapes. The model derivative shape is ",
                    exact_val.sizes(),
                    " while the finite differencing derivative shape is ",
                    numerical.sizes(),
                    ".");
        neml_assert(
            at::allclose(exact_val, numerical, _deriv_rtol, _deriv_atol),
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
}

void
ModelUnitTest::check_d2value()
{
  _model->clear_input();
  _model->assign_input(_in);
  _model->zero_undefined_input();
  _model->forward_maybe_jit(false, false, true);
  const auto exact = _model->collect_output_second_derivatives();

  for (const auto & [yname, yvar] : _model->output_variables())
    for (const auto & [x1name, x1var] : _model->input_variables())
      for (auto & [x2name, x2var] : _model->input_variables())
      {
        _model->clear_input();
        _model->assign_input(_in);
        _model->zero_undefined_input();

        const auto & x20 = x2var->tensor();
        auto numerical = finite_differencing_derivative(
            [this, &yvar = yvar, &x1var = x1var, &x2var = x2var](const Tensor & x)
            {
              (*x2var) = x;
              _model->forward_maybe_jit(false, true, false);
              if (!yvar->has_derivative(x1var->name()))
                return Tensor::zeros(utils::add_shapes(yvar->base_sizes(), x1var->base_sizes()),
                                     x.options());
              return yvar->d(*x1var).tensor();
            },
            x20,
            1e-6,
            1e-6);

        // If the derivative does not exist, the numerical derivative should be zero
        if (!exact.count(yname) || !exact.at(yname).count(x1name) ||
            !exact.at(yname).at(x1name).count(x2name))
          neml_assert(
              at::allclose(numerical, at::zeros_like(numerical), _secderiv_rtol, _secderiv_atol),
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
              at::allclose(
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
  for (auto && [name, param] : _model->named_parameters())
    param->requires_grad_(true);

  // Evaluate the model
  _model->clear_input();
  _model->assign_input(_in);
  _model->zero_undefined_input();
  _model->forward_maybe_jit(true, false, false);

  // Extract AD parameter derivatives
  std::map<VariableName, std::map<std::string, Tensor>> exact;
  for (const auto & [yname, yvar] : _model->output_variables())
  {
    for (auto && [pname, param] : _model->named_parameters())
    {
      auto deriv = jacrev(yvar->tensor(),
                          Tensor(*param),
                          /*retain_graph=*/true,
                          /*create_graph=*/false,
                          /*allow_unused=*/true);
      if (deriv.defined())
        exact[yname][pname] = deriv;
    }
  }

  // Compare results against FD
  for (const auto & [yname, yvar] : _model->output_variables())
  {
    for (auto && [pname, param] : _model->named_parameters())
    {
      auto numerical = finite_differencing_derivative(
          [&, &pname = pname, &param = param, &yvar = yvar](const Tensor & x)
          {
            auto p0 = Tensor(*param).clone();
            _model->set_parameter(pname, x);
            _model->forward_maybe_jit(true, false, false);
            _model->set_parameter(pname, p0);
            return yvar->tensor();
          },
          Tensor(*param),
          1e-6,
          1e-6);
      if (exact.count(yname) && exact[yname].count(pname))
        neml_assert(at::allclose(exact[yname][pname], numerical, _param_rtol, _param_atol),
                    "The model gives derivative of output variable '",
                    yname,
                    "' w.r.t. parameter '",
                    pname,
                    "' different from those given by finite differencing. The model gives:\n",
                    exact[yname][pname],
                    "\nFinite differencing gives:\n",
                    numerical);
      else
        neml_assert(at::allclose(numerical, at::zeros_like(numerical), _param_rtol, _param_atol),
                    "The model gives zero derivative of output variable '",
                    yname,
                    "' w.r.t. parameter '",
                    pname,
                    "', but finite differencing gives:\n",
                    numerical);
    }
  }
}

} // namespace neml2
