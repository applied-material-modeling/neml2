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

#include "neml2/drivers/TransientDriver.h"
#include "neml2/models/ComposedModel.h"
#include "neml2/models/ImplicitUpdate.h"

namespace fs = std::filesystem;

namespace neml2
{
OptionSet
TransientDriver::expected_options()
{
  OptionSet options = Driver::expected_options();

  options.set<std::string>("model");
  options.set("model").doc() = "The material model to be updated by the driver";

  options.set<VariableName>("time") = VariableName("forces", "t");
  options.set("time").doc() = "Time";
  options.set<CrossRef<Scalar>>("prescribed_time");
  options.set("prescribed_time").doc() =
      "Time steps to perform the material update. The times tensor must "
      "have at least one batch dimension representing time steps";

  EnumSelection predictor_selection({"PREVIOUS_STATE", "LINEAR_EXTRAPOLATION"}, "PREVIOUS_STATE");
  options.set<EnumSelection>("predictor") = predictor_selection;
  options.set("predictor").doc() =
      "Predictor used to set the initial guess for each time step. Options are " +
      predictor_selection.candidates_str();

  options.set<std::string>("save_as");
  options.set("save_as").doc() =
      "File path (absolute or relative to the working directory) to store the results";

  options.set<bool>("show_parameters") = false;
  options.set("show_parameters").doc() = "Whether to show model parameters at the beginning";
  options.set<bool>("show_input_axis") = false;
  options.set("show_input_axis").doc() = "Whether to show model input axis at the beginning";
  options.set<bool>("show_output_axis") = false;
  options.set("show_output_axis").doc() = "Whether to show model output axis at the beginning";

  options.set<std::string>("device") = "cpu";
  options.set("device").doc() =
      "Device on which to evaluate the material model. The string supplied must follow the "
      "following schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies the device type, "
      "and :<device-index> optionally specifies a device index. For example, device='cpu' sets the "
      "target compute device to be CPU, and device='cuda:1' sets the target compute device to be "
      "CUDA with device ID 1.";

  options.set<std::vector<VariableName>>("ic_scalar_names");
  options.set("ic_scalar_names").doc() = "Apply initial conditions to these Scalar variables";
  options.set<std::vector<CrossRef<Scalar>>>("ic_scalar_values");
  options.set("ic_scalar_values").doc() = "Initial condition values for the Scalar variables";

  options.set<std::vector<VariableName>>("ic_rot_names");
  options.set("ic_rot_names").doc() = "Apply initial conditions to these Rot variables";
  options.set<std::vector<CrossRef<Rot>>>("ic_rot_values");
  options.set("ic_rot_values").doc() = "Initial condition values for the Rot variables";

  options.set<std::vector<VariableName>>("ic_sr2_names");
  options.set("ic_sr2_names").doc() = "Apply initial conditions to these SR2 variables";
  options.set<std::vector<CrossRef<SR2>>>("ic_sr2_values");
  options.set("ic_sr2_values").doc() = "Initial condition values for the SR2 variables";

  return options;
}

TransientDriver::TransientDriver(const OptionSet & options)
  : Driver(options),
    _model(get_model(options.get<std::string>("model"))),
    _device(options.get<std::string>("device")),
    _time_name(options.get<VariableName>("time")),
    _time(options.get<CrossRef<Scalar>>("prescribed_time")),
    _step_count(0),
    _nsteps(_time.batch_size(0).concrete()),
    _predictor(options.get<EnumSelection>("predictor")),
    _save_as(options.get<std::string>("save_as")),
    _show_params(options.get<bool>("show_parameters")),
    _show_input(options.get<bool>("show_input_axis")),
    _show_output(options.get<bool>("show_output_axis")),
    _result_in(_nsteps),
    _result_out(_nsteps)
{
  _time = _time.to(_device);
}

void
TransientDriver::diagnose(std::vector<Diagnosis> & diagnoses) const
{
  Driver::diagnose(diagnoses);
  _model.diagnose(diagnoses);
  diagnostic_assert(
      diagnoses,
      _time.batch_dim() >= 1,
      "Input time should have at least one batch dimension but instead has batch dimension ",
      _time.batch_dim());
}

bool
TransientDriver::run()
{
  // LCOV_EXCL_START
  if (_show_params)
    for (auto && [pname, pval] : _model.named_parameters())
      std::cout << pname << std::endl;

  if (_show_input)
    std::cout << _model.name() << "'s input axis:\n" << _model.input_axis() << std::endl;

  if (_show_output)
    std::cout << _model.name() << "'s output axis:\n" << _model.output_axis() << std::endl;
  // LCOV_EXCL_STOP

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
  if (_model.input_axis().has_old_state())
  {
    const auto input_old_state = _model.input_axis().subaxis("old_state");
    for (const auto & var : input_old_state.variable_names())
      _in[var.prepend("old_state")] = _result_out[_step_count - 1][var.prepend("state")];
  }

  // Forces from the previous time step become the old forces in the current time step
  if (_model.input_axis().has_old_forces())
  {
    const auto input_old_forces = _model.input_axis().subaxis("old_forces");
    for (const auto & var : input_old_forces.variable_names())
      _in[var.prepend("old_forces")] = _result_in[_step_count - 1][var.prepend("forces")];
  }
}

void
TransientDriver::update_forces()
{
  if (_model.input_axis().has_variable(_time_name))
    _in[_time_name] = _time.batch_index({_step_count});
}

void
TransientDriver::apply_ic()
{
  set_ic<Scalar>("ic_scalar_names", "ic_scalar_values");
  set_ic<Rot>("ic_rot_names", "ic_rot_values");
  set_ic<SR2>("ic_sr2_names", "ic_sr2_values");

  for (auto && [name, var] : _model.output_variables())
    if (!_result_out[0].count(name))
      _result_out[0][name] =
          Tensor::zeros(utils::add_shapes(var.list_sizes(), var.base_sizes())).to(_device);
}

void
TransientDriver::apply_predictor()
{
  if (!_model.input_axis().has_state())
    return;

  const auto input_state = _model.input_axis().subaxis("state");
  for (const auto & var : input_state.variable_names())
    if (_model.output_axis().has_variable(var.prepend("state")))
    {
      if (_predictor == "PREVIOUS_STATE")
        _in[var.prepend("state")] = _result_out[_step_count - 1][var.prepend("state")];
      else if (_predictor == "LINEAR_EXTRAPOLATION")
      {
        // Fall back to PREVIOUS_STATE predictor at the 1st time step
        if (_step_count == 1)
          _in[var.prepend("state")] = _result_out[_step_count - 1][var.prepend("state")];
        // Otherwise linearly extrapolate in time
        else
        {
          const auto t = Scalar(_in[_time_name]);
          const auto t_n = Scalar(_result_in[_step_count - 1][_time_name]);
          const auto t_nm1 = Scalar(_result_in[_step_count - 2][_time_name]);
          const auto dt = t - t_n;
          const auto dt_n = t_n - t_nm1;

          const auto s_n = _result_out[_step_count - 1][var.prepend("state")];
          const auto s_nm1 = _result_out[_step_count - 2][var.prepend("state")];
          _in[var.prepend("state")] = s_n + (s_n - s_nm1) / dt_n * dt;
        }
      }
      else
        throw NEMLException("Unrecognized predictor type: " + std::string(_predictor));
    }
}

void
TransientDriver::solve_step()
{
  _result_out[_step_count] = _model.value(_in);
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
