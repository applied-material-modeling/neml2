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
  neml_assert(factory, "Internal error: factory == nullptr");
  for (std::size_t i = 0; i < names.size(); i++)
  {
    neml_assert(names[i].is_state(),
                "Initial condition names should start with 'state' but instead got ",
                names[i]);
    storage[names[i]] = vals[i].resolve(factory).to(device);
  }
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
  neml_assert(factory, "Internal error: factory == nullptr");
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
      predictor_selection.join();

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
    _time(resolve_tensor<Scalar>("prescribed_time")),
    _nsteps(_time.dynamic_size(0).concrete()),
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

  for (const auto & [vname, var] : _model->input_variables())
    if (var->is_state())
    {
      _has_input_state = true;
      break;
    }
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

  // Check for statefulness
  bool has_old_state = false;
  for (const auto & [vname, var] : _model->input_variables())
    if (var->is_old_state())
    {
      has_old_state = true;
      break;
    }

  if (has_old_state)
    for (const auto & [vname, var] : _model->input_variables())
      if (var->is_old_state())
        diagnostic_assert(_model->output_variables().count(vname.remount(STATE)),
                          "Input has old state variable ",
                          vname,
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
  // State from the previous time step becomes the old state in the current time step
  for (const auto & [var, val] : _result_out[_step_count - 1])
    if (var.is_state() && _model->input_variables().count(var.remount(OLD_STATE)))
      _in[var.remount(OLD_STATE)] = val;

  // Forces from the previous time step become the old forces in the current time step
  for (const auto & [var, val] : _result_in[_step_count - 1])
    if (var.is_force() && _model->input_variables().count(var.remount(OLD_FORCES)))
      _in[var.remount(OLD_FORCES)] = val;
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
  _result_out[0] = _ics;

  // Figure out what the batch size for our default zero ICs should be
  std::vector<Tensor> defined;
  for (const auto & [vname, var] : _model->output_variables())
    if (_result_out[0].count(vname))
      defined.push_back(_result_out[0][vname]);
  for (const auto & [key, value] : _in)
    defined.push_back(value);
  const auto dynamic_shape = utils::broadcast_dynamic_sizes(defined);

  // Variables without a user-defined IC are initialized to zeros
  for (auto && [name, var] : _model->output_variables())
    if (!_result_out[0].count(name))
      _result_out[0][name] = Tensor::zeros(dynamic_shape, {}, var->base_sizes(), _device);
}

void
TransientDriver::apply_predictor()
{
  if (!_has_input_state)
    return;

  for (const auto & [vname, var] : _model->input_variables())
    if (vname.is_state())
      if (_model->output_variables().count(vname.remount(STATE)))
      {
        if (_predictor == "PREVIOUS_STATE")
          _in[vname.remount(STATE)] = _result_out[_step_count - 1][vname.remount(STATE)];
        else if (_predictor == "LINEAR_EXTRAPOLATION")
        {
          // Fall back to PREVIOUS_STATE predictor at the 1st time step
          if (_step_count == 1)
            _in[vname.remount(STATE)] = _result_out[_step_count - 1][vname.remount(STATE)];
          // Otherwise linearly extrapolate in time
          else
          {
            const auto t = Scalar(_in[_time_name]);
            const auto t_n = Scalar(_result_in[_step_count - 1][_time_name]);
            const auto t_nm1 = Scalar(_result_in[_step_count - 2][_time_name]);
            const auto dt = t - t_n;
            const auto dt_n = t_n - t_nm1;

            const auto s_n = _result_out[_step_count - 1][vname.remount(STATE)];
            const auto s_nm1 = _result_out[_step_count - 2][vname.remount(STATE)];
            _in[vname.remount(STATE)] = s_n + (s_n - s_nm1) / dt_n * dt;
          }
        }
        else
          throw NEMLException("Unrecognized predictor type: " + _predictor.selection());
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
