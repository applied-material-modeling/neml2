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

#pragma once

#include "neml2/drivers/Driver.h"
#include "neml2/tensors/tensors.h"

#include <filesystem>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/moduledict.h>
#include <torch/serialize.h>

namespace neml2
{
/**
 * @brief The driver for a transient initial-value problem.
 *
 */
class TransientDriver : public Driver
{
public:
  static OptionSet expected_options();

  /**
   * @brief Construct a new TransientDriver object
   *
   * @param options The options extracted from the input file
   */
  TransientDriver(const OptionSet & options);

  virtual void diagnose(std::vector<Diagnosis> &) const override;

  bool run() override;

  const Model & model() const { return _model; }

  /// The destination file/path to save the results.
  virtual std::string save_as_path() const;

  /**
   * @brief The results (input and output) from all time steps.
   *
   * @return torch::nn::ModuleDict The results (input and output) from all time steps. The keys of
   * the dict are "input" and "output". The values are torch::nn::ModuleList with length equal to
   * the number of time steps. Each ModuleList contains the input/output variables at each time step
   * in the form of torch::nn::ModuleDict whose keys are variable names and values are variable
   * values.
   */
  virtual torch::nn::ModuleDict result() const;

protected:
  /// Solve the initial value problem
  virtual bool solve();

  // @{ Routines that are called every step
  /// Advance in time: the state becomes old state, and forces become old forces.
  virtual void advance_step();
  /// Update the driving forces for the current time step.
  virtual void update_forces();
  /// Apply the initial conditions.
  virtual void apply_ic();
  /// Apply the predictor to calculate the initial guess for the current time step.
  virtual void apply_predictor();
  /// Perform the constitutive update for the current time step.
  virtual void solve_step();
  /// Save the input of the current time step.
  virtual void store_input();
  // @}

  /// Save the results into the destination file/path.
  virtual void output() const;

  /// The model which the driver uses to perform constitutive updates.
  Model & _model;
  /// The device on which to evaluate the model
  const torch::Device _device;

  /// VariableName for the time
  const VariableName _time_name;
  /// The current time
  Scalar _time;
  /// The current step count
  Size _step_count;
  /// Total number of steps
  const Size _nsteps;
  /// The input to the constitutive model
  std::map<VariableName, Tensor> _in;

  /// The predictor used to set the initial guess
  const EnumSelection _predictor;

  /// The destination file name or file path
  std::string _save_as;
  /// Set to true to list all the model parameters at the beginning
  const bool _show_params;
  /// Set to true to show model's input axis at the beginning
  const bool _show_input;
  /// Set to true to show model's output axis at the beginning
  const bool _show_output;

  /// Inputs from all time steps
  std::vector<std::map<VariableName, Tensor>> _result_in;
  /// Outputs from all time steps
  std::vector<std::map<VariableName, Tensor>> _result_out;

private:
  void output_pt(const std::filesystem::path & out) const;

  template <typename T>
  void set_ic(const std::string & name_opt, const std::string & value_opt)
  {
    const auto names = input_options().get<std::vector<VariableName>>(name_opt);
    const auto vals = input_options().get<std::vector<CrossRef<T>>>(value_opt);
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
    for (std::size_t i = 0; i < names.size(); i++)
    {
      neml_assert(names[i].start_with("state"),
                  "Initial condition names should start with 'state' but instead got ",
                  names[i]);
      _result_out[0][names[i]] = T(vals[i]).to(_device);
    }
  }
};
} // namespace neml2
