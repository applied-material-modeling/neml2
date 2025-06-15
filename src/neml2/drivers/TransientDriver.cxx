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

#ifdef NEML2_HAS_DISPATCHER
#include "neml2/dispatchers/ValueMapLoader.h"
#endif

namespace fs = std::filesystem;

namespace neml2
{
register_NEML2_object(TransientDriver);

template <typename T>
void
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
  neml_assert(factory, "Internal error: factory != nullptr");
  for (std::size_t i = 0; i < names.size(); i++)
  {
    neml_assert(names[i].is_state(),
                "Initial condition names should start with 'state' but instead got ",
                names[i]);
    storage[names[i]] = vals[i].resolve(factory).to(device);
  }
}

template <typename T>
void
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
  neml_assert(factory, "Internal error: factory != nullptr");
  for (std::size_t i = 0; i < force_names.size(); i++)
  {
    neml_assert(force_names[i].is_force(),
                "Driving force names should start with 'forces' but instead got ",
                force_names[i]);
    names.push_back(force_names[i]);
    values.push_back(vals[i].resolve(factory).to(device));
  }
}

OptionSet
TransientDriver::expected_options()
{
  OptionSet options = ModelDriver::expected_options();
  options.doc() = "Driver for simulating the transient response of an autonomous system.";

  options.set<VariableName>("time") = VariableName(FORCES, "t");
  options.set("time").doc() = "Time";
  options.set<TensorName<Scalar>>("prescribed_time");
  options.set("prescribed_time").doc() =
      "Time steps to perform the material update. The times tensor must "
      "have at least one batch dimension representing time steps";

  EnumSelection predictor_selection({"PREVIOUS_STATE", "LINEAR_EXTRAPOLATION"}, "PREVIOUS_STATE");
  options.set<EnumSelection>("predictor") = predictor_selection;
  options.set("predictor").doc() =
      "Predictor used to set the initial guess for each time step. Options are " +
      predictor_selection.candidates_str();

#define OPTION_IC_(T)                                                                              \
  options.set<std::vector<VariableName>>("ic_" #T "_names");                                       \
  options.set("ic_" #T "_names").doc() = "Apply initial conditions to these " #T " variables";     \
  options.set<std::vector<TensorName<T>>>("ic_" #T "_values");                                     \
  options.set("ic_" #T "_values").doc() = "Initial condition values for the " #T " variables"
  FOR_ALL_TENSORBASE(OPTION_IC_);

#define OPTION_FORCE_(T)                                                                           \
  options.set<std::vector<VariableName>>("force_" #T "_names");                                    \
  options.set("force_" #T "_names").doc() = "Prescribed driving force of tensor type " #T;         \
  options.set<std::vector<TensorName<T>>>("force_" #T "_values");                                  \
  options.set("force_" #T "_values").doc() = "Prescribed driving force values of tensor type " #T
  FOR_ALL_TENSORBASE(OPTION_FORCE_);

  options.set<std::string>("save_as");
  options.set("save_as").doc() =
      "File path (absolute or relative to the working directory) to store the results";

  return options;
}

TransientDriver::TransientDriver(const OptionSet & options)
  : ModelDriver(options),
    _time_name(options.get<VariableName>("time")),
    _time(options.get<TensorName<Scalar>>("prescribed_time").resolve(factory())),
    _nsteps(_time.batch_size(0).concrete()),
    _predictor(options.get<EnumSelection>("predictor")),
    _result_in(_nsteps),
    _result_out(_nsteps),
    _save_as(options.get<std::string>("save_as"))
{
  _time = _time.to(_device);

#define SET_IC_(T) set_ic<T>(_ics, options, "ic_" #T "_names", "ic_" #T "_values", _device)
  FOR_ALL_TENSORBASE(SET_IC_);

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
      _time.batch_dim() >= 1,
      "Input time should have at least one batch dimension but instead has batch dimension ",
      _time.batch_dim());

  for (std::size_t i = 0; i < _driving_forces.size(); i++)
  {
    diagnostic_assert(_driving_forces[i].batch_dim() >= 1,
                      "Input driving force ",
                      _driving_force_names[i],
                      " should have at least one batch dimension but instead has batch dimension ",
                      _driving_forces[i].batch_dim());
    diagnostic_assert(_driving_forces[i].batch_size(0) == _time.batch_size(0),
                      "Prescribed driving force ",
                      _driving_force_names[i],
                      " should have the same number of steps "
                      "as time, but instead has ",
                      _driving_forces[i].batch_size(0),
                      " steps");
  }

  // Check for statefulness
  const auto & input_old_state = _model->input_axis().subaxis(OLD_STATE);
  const auto & output_state = _model->output_axis().subaxis(STATE);
  if (_model->input_axis().has_old_state())
    for (const auto & var : input_old_state.variable_names())
      diagnostic_assert(output_state.has_variable(var),
                        "Input axis has old state variable ",
                        var,
                        ", but the corresponding output state variable doesn't exist.");
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
  // State from the previous time step becomes the old state in the current time step
  if (_model->input_axis().has_old_state())
  {
    const auto input_old_state = _model->input_axis().subaxis(OLD_STATE);
    for (const auto & var : input_old_state.variable_names())
      _in[var.prepend(OLD_STATE)] = _result_out[_step_count - 1][var.prepend(STATE)];
  }

  // Forces from the previous time step become the old forces in the current time step
  if (_model->input_axis().has_old_forces())
  {
    const auto input_old_forces = _model->input_axis().subaxis(OLD_FORCES);
    for (const auto & var : input_old_forces.variable_names())
      _in[var.prepend(OLD_FORCES)] = _result_in[_step_count - 1][var.prepend(FORCES)];
  }
}

void
TransientDriver::update_forces()
{
  if (_model->input_axis().has_variable(_time_name))
    _in[_time_name] = _time.batch_index({_step_count});

  for (std::size_t i = 0; i < _driving_force_names.size(); i++)
    _in[_driving_force_names[i]] = _driving_forces[i].batch_index({_step_count});
}

void
TransientDriver::apply_ic()
{
  _result_out[0] = _ics;

  // Figure out what the batch size for our default zero ICs should be
  std::vector<Tensor> defined;
  for (const auto & var : _model->output_axis().variable_names())
    if (_result_out[0].count(var))
      defined.push_back(_result_out[0][var]);
  for (const auto & [key, value] : _in)
    defined.push_back(value);
  const auto batch_shape = utils::broadcast_batch_sizes(defined);

  // Variables without a user-defined IC are initialized to zeros
  for (auto && [name, var] : _model->output_variables())
    if (!_result_out[0].count(name))
    {
      if (batch_shape.size() > 0)
        _result_out[0][name] =
            Tensor::zeros(utils::add_shapes(var->list_sizes(), var->base_sizes()))
                .to(_device)
                .batch_unsqueeze(0)
                .batch_expand(batch_shape);
      else
        _result_out[0][name] =
            Tensor::zeros(utils::add_shapes(var->list_sizes(), var->base_sizes())).to(_device);
    }
}

void
TransientDriver::apply_predictor()
{
  if (!_model->input_axis().has_state())
    return;

  const auto input_state = _model->input_axis().subaxis(STATE);
  for (const auto & var : input_state.variable_names())
    if (_model->output_axis().has_variable(var.prepend(STATE)))
    {
      if (_predictor == "PREVIOUS_STATE")
        _in[var.prepend(STATE)] = _result_out[_step_count - 1][var.prepend(STATE)];
      else if (_predictor == "LINEAR_EXTRAPOLATION")
      {
        // Fall back to PREVIOUS_STATE predictor at the 1st time step
        if (_step_count == 1)
          _in[var.prepend(STATE)] = _result_out[_step_count - 1][var.prepend(STATE)];
        // Otherwise linearly extrapolate in time
        else
        {
          const auto t = Scalar(_in[_time_name]);
          const auto t_n = Scalar(_result_in[_step_count - 1][_time_name]);
          const auto t_nm1 = Scalar(_result_in[_step_count - 2][_time_name]);
          const auto dt = t - t_n;
          const auto dt_n = t_n - t_nm1;

          const auto s_n = _result_out[_step_count - 1][var.prepend(STATE)];
          const auto s_nm1 = _result_out[_step_count - 2][var.prepend(STATE)];
          _in[var.prepend(STATE)] = s_n + (s_n - s_nm1) / dt_n * dt;
        }
      }
      else
        throw NEMLException("Unrecognized predictor type: " + std::string(_predictor));
    }
}

void
TransientDriver::solve_step()
{
#ifdef NEML2_HAS_DISPATCHER
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
