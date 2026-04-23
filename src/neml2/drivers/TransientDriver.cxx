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

#include <torch/nn/modules/container/moduledict.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/serialize.h>

#include "neml2/drivers/TransientDriver.h"
#include "neml2/misc/assertions.h"
#include "neml2/models/Model.h"
#include "neml2/base/Settings.h"

#ifdef NEML2_WORK_DISPATCHER
#include "neml2/dispatchers/ValueMapLoader.h"
#endif

namespace fs = std::filesystem;

namespace neml2
{
register_NEML2_object(TransientDriver);

template <typename T>
static void
set_ic(ValueMap & storage,
       const OptionSet & options,
       const std::string & name_opt,
       const std::string & value_opt,
       const Device & device)
{
  const auto & names = options.get<std::vector<VariableName>>(name_opt);
  const auto & vals = options.get<std::vector<TensorName<T>>>(value_opt);
  neml_assert(names.size() == vals.size(),
              "Number of initial condition names ",
              name_opt,
              " and number of initial condition values ",
              value_opt,
              " should be the same but instead have ",
              names.size(),
              " and ",
              vals.size(),
              " respectively.");
  auto * factory = options.get<Factory *>("_factory");
  neml_assert(factory,
              "Internal error: factory is null while resolving tensor names. Ensure this driver "
              "is created via the NEML2 factory.");
  for (std::size_t i = 0; i < names.size(); i++)
    storage[names[i]] = vals[i].resolve(factory).to(device);
}

template <typename T>
static void
get_force(std::vector<VariableName> & names,
          std::vector<Tensor> & values,
          const OptionSet & options,
          const std::string & name_opt,
          const std::string & value_opt,
          const Device & device)
{
  const auto & force_names = options.get<std::vector<VariableName>>(name_opt);
  const auto & vals = options.get<std::vector<TensorName<T>>>(value_opt);
  neml_assert(force_names.size() == vals.size(),
              "Number of driving force names ",
              name_opt,
              " and number of driving force values ",
              value_opt,
              " should be the same but instead have ",
              force_names.size(),
              " and ",
              vals.size(),
              " respectively.");
  auto * factory = options.get<Factory *>("_factory");
  neml_assert(factory,
              "Internal error: factory is null while resolving tensor names. Ensure this driver "
              "is created via the NEML2 factory.");
  for (std::size_t i = 0; i < force_names.size(); i++)
  {
    names.push_back(force_names[i]);
    values.push_back(vals[i].resolve(factory).to(device));
  }
}

OptionSet
TransientDriver::expected_options()
{
  OptionSet options = ModelDriver::expected_options();
  options.doc() = "Driver for simulating the transient response of an autonomous system.";

  options.add<VariableName>("time", "t", "Time");
  options.add<TensorName<Scalar>>(
      "prescribed_time",
      "Time steps to perform the material update. The times tensor must have at least one batch "
      "dimension representing time steps");

  EnumSelection predictor_selection({"PREVIOUS_STATE", "LINEAR_EXTRAPOLATION"}, "PREVIOUS_STATE");
  options.add<EnumSelection>(
      "predictor",
      predictor_selection,
      "Predictor used to set the initial guess for each time step. Options are " +
          predictor_selection.join());

#define OPTION_IC_(T)                                                                              \
  options.add<std::vector<VariableName>>(                                                          \
      "ic_" #T "_names", {}, "Apply initial conditions to these " #T " variables");                \
  options.add<std::vector<TensorName<T>>>(                                                         \
      "ic_" #T "_values", {}, "Initial condition values for the " #T " variables");
  FOR_ALL_TENSORBASE(OPTION_IC_);

#define OPTION_FORCE_(T)                                                                           \
  options.add<std::vector<VariableName>>(                                                          \
      "force_" #T "_names", {}, "Prescribed driving force of tensor type " #T);                    \
  options.add<std::vector<TensorName<T>>>(                                                         \
      "force_" #T "_values", {}, "Prescribed driving force values of tensor type " #T);
  FOR_ALL_TENSORBASE(OPTION_FORCE_);

  options.add_optional<std::string>(
      "save_as", "File path (absolute or relative to the working directory) to store the results");

  options.add_optional<std::string>(
      "custom_predictor",
      "Name of a model to use as an additional predictor before each solve step. Its outputs "
      "override state variable initial guesses.");

  EnumSelection custom_predictor_apply_selection({"FIRST_STEP", "ALL_STEPS"}, "FIRST_STEP");
  options.add<EnumSelection>(
      "custom_predictor_apply",
      custom_predictor_apply_selection,
      "When to apply the custom predictor: FIRST_STEP (only the first step) or ALL_STEPS (every "
      "step). Options are " +
          custom_predictor_apply_selection.join());

  return options;
}

TransientDriver::TransientDriver(const OptionSet & options)
  : ModelDriver(options),
    _time_name(options.get<VariableName>("time")),
    _time(resolve_tensor<Scalar>("prescribed_time")),
    _nsteps(_time.dynamic_size(0).concrete()),
    _predictor(options.get<EnumSelection>("predictor")),
    _custom_predictor_model(options.defined("custom_predictor")
                                ? factory()->get_model(options.get<std::string>("custom_predictor"))
                                : nullptr),
    _custom_predictor_apply(options.get<EnumSelection>("custom_predictor_apply")),
    _result_in(_nsteps),
    _result_out(_nsteps),
    _save_as(options.defined("save_as") ? options.get<std::string>("save_as") : "")
{
  _time = _time.to(_device);

#define GET_FORCE_(T)                                                                              \
  get_force<T>(_driving_force_names,                                                               \
               _driving_forces,                                                                    \
               options,                                                                            \
               "force_" #T "_names",                                                               \
               "force_" #T "_values",                                                              \
               _device)
  FOR_ALL_TENSORBASE(GET_FORCE_);
}

void
TransientDriver::setup()
{
  ModelDriver::setup();
}

void
TransientDriver::diagnose() const
{
  ModelDriver::diagnose();

  diagnostic_assert(
      _time.dynamic_dim() >= 1,
      "Input time should have at least one batch dimension but instead has batch dimension ",
      _time.dynamic_dim());

  for (std::size_t i = 0; i < _driving_forces.size(); i++)
  {
    diagnostic_assert(_driving_forces[i].dynamic_dim() >= 1,
                      "Input driving force ",
                      _driving_force_names[i],
                      " should have at least one batch dimension but instead has batch dimension ",
                      _driving_forces[i].dynamic_dim());
    diagnostic_assert(_driving_forces[i].dynamic_size(0) == _time.dynamic_size(0),
                      "Prescribed driving force ",
                      _driving_force_names[i],
                      " should have the same number of steps "
                      "as time, but instead has ",
                      _driving_forces[i].dynamic_size(0),
                      " steps");
  }
}

bool
TransientDriver::run()
{
  auto status = solve();

  if (!save_as_path().empty())
    output();

  return status;
}

bool
TransientDriver::solve()
{
  for (_step_count = 0; _step_count < _nsteps; _step_count++)
  {
    if (_verbose)
      // LCOV_EXCL_START
      std::cout << "Step " << _step_count << std::endl;
    // LCOV_EXCL_STOP

    if (_step_count > 0)
      advance_step();
    update_forces();
    if (_step_count == 0)
    {
      store_input();
      apply_ic();
    }
    else
    {
      apply_predictor();
      store_input();
      solve_step();
      postprocess();
    }

    if (_verbose)
      // LCOV_EXCL_START
      std::cout << std::endl;
    // LCOV_EXCL_STOP
  }

  return true;
}

void
TransientDriver::advance_step()
{
  const auto & prev_out = _result_out[_step_count - 1];
  const auto & prev_in = _result_in[_step_count - 1];

  for (const auto & [vname, var] : _model->input_variables())
  {
    const auto order = var->history_order();
    if (order == 0)
      continue;

    const auto & base = var->base_name();
    if (order == 1)
    {
      // Prefer model output (state variables) over input (forces like time)
      if (prev_out.count(base))
        _in[vname] = prev_out.at(base);
      else if (prev_in.count(base))
        _in[vname] = prev_in.at(base);
    }
    else
    {
      // For higher-order history, look up the lower-order history from the previous step's input
      const auto & sep = _model->settings().history_separator();
      const auto prev_history_name = base + sep + std::to_string(order - 1);
      if (prev_in.count(prev_history_name))
        _in[vname] = prev_in.at(prev_history_name);
    }
  }
}

void
TransientDriver::update_forces()
{
  if (_model->input_variables().count(_time_name))
    _in[_time_name] = _time.dynamic_index({_step_count});

  for (std::size_t i = 0; i < _driving_force_names.size(); i++)
    _in[_driving_force_names[i]] = _driving_forces[i].dynamic_index({_step_count});
}

void
TransientDriver::apply_ic()
{
#define SET_IC_(T)                                                                                 \
  set_ic<T>(_result_out[0], input_options(), "ic_" #T "_names", "ic_" #T "_values", _device)
  FOR_ALL_TENSORBASE(SET_IC_);

  // Variables without a user-defined IC are initialized to zeros
  for (auto && [name, var] : _model->output_variables())
    if (!_result_out[0].count(name))
      _result_out[0][name] = var->zeros(_device);
}

void
TransientDriver::apply_predictor()
{
  for (const auto & [vname, var] : _model->input_variables())
    if (_model->output_variables().count(vname))
    {
      if (_predictor == "PREVIOUS_STATE")
        _in[vname] = _result_out[_step_count - 1][vname];
      else if (_predictor == "LINEAR_EXTRAPOLATION")
      {
        // Fall back to PREVIOUS_STATE predictor at the 1st time step
        if (_step_count == 1)
          _in[vname] = _result_out[_step_count - 1][vname];
        // Otherwise linearly extrapolate in time
        else
        {
          const auto t = Scalar(_in[_time_name]);
          const auto t_n = Scalar(_result_in[_step_count - 1][_time_name]);
          const auto t_nm1 = Scalar(_result_in[_step_count - 2][_time_name]);
          const auto dt = t - t_n;
          const auto dt_n = t_n - t_nm1;

          const auto s_n = _result_out[_step_count - 1][vname];
          const auto s_nm1 = _result_out[_step_count - 2][vname];
          _in[vname] = s_n + (s_n - s_nm1) / dt_n * dt;
        }
      }
      else
        throw NEMLException("Unrecognized predictor type: " + _predictor.selection());
    }

  if (_custom_predictor_model && (_custom_predictor_apply == "ALL_STEPS" || _step_count == 1))
  {
    // Build a filtered input map for the custom predictor (only its declared inputs)
    ValueMap predictor_in;
    for (const auto & [vname, var] : _custom_predictor_model->input_variables())
      if (_in.count(vname))
        predictor_in[vname] = _in.at(vname);

    const auto predictor_out = _custom_predictor_model->value(predictor_in);
    for (const auto & [vname, val] : predictor_out)
      if (_model->input_variables().count(vname))
        _in[vname] = val;
  }
}

void
TransientDriver::solve_step()
{
#ifdef NEML2_WORK_DISPATCHER
  if (_dispatcher)
  {
    ValueMapLoader loader(_in, 0);
    _result_out[_step_count] = _dispatcher->run(loader);
    return;
  }
#endif

  _result_out[_step_count] = _model->value((_in));
}

void
TransientDriver::postprocess()
{
  if (!_postprocessor)
    return;

  ValueMap pp_in;
  for (const auto & [name, var] : _postprocessor->input_variables())
  {
    neml_assert(_result_out[_step_count].count(name),
                "Postprocessor input variable ",
                name,
                " not found in model output.");
    pp_in[name] = _result_out[_step_count][name];
  }

  const auto pp_out = _postprocessor->value(pp_in);
  for (const auto & [name, val] : pp_out)
    _result_out[_step_count][name] = val;
}

void
TransientDriver::store_input()
{
  _result_in[_step_count] = _in;
}

std::string
TransientDriver::save_as_path() const
{
  return _save_as;
}

torch::nn::ModuleDict
TransientDriver::result() const
{
  // Dump input variables into a ModuleList
  torch::nn::ModuleList res_in;
  for (const auto & in : _result_in)
  {
    // Dump input variables at each step into a ModuleDict
    torch::nn::ModuleDict res_in_step;
    for (auto && [name, val] : in)
      res_in_step->register_buffer(utils::stringify(name), val);
    res_in->push_back(res_in_step);
  }

  // Dump output variables into a ModuleList
  torch::nn::ModuleList res_out;
  for (const auto & out : _result_out)
  {
    // Dump output variables at each step into a ModuleDict
    torch::nn::ModuleDict res_out_step;
    for (auto && [name, val] : out)
      res_out_step->register_buffer(utils::stringify(name), val);
    res_out->push_back(res_out_step);
  }

  // Combine input and output
  torch::nn::ModuleDict res;
  res->update({{"input", res_in.ptr()}, {"output", res_out.ptr()}});
  return res;
}

void
TransientDriver::output() const
{
  if (_verbose)
    // LCOV_EXCL_START
    std::cout << "Saving results..." << std::endl;
  // LCOV_EXCL_STOP

  auto cwd = fs::current_path();
  auto out = cwd / save_as_path();

  if (out.extension() == ".pt")
    output_pt(out);
  else
    // LCOV_EXCL_START
    neml_assert(false, "Unsupported output format: ", out.extension());
  // LCOV_EXCL_STOP

  if (_verbose)
    // LCOV_EXCL_START
    std::cout << "Results saved to " << save_as_path() << std::endl;
  // LCOV_EXCL_STOP
}

void
TransientDriver::output_pt(const std::filesystem::path & out) const
{
  torch::save(result(), out);
}
} // namespace neml2
