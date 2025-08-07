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

#include <filesystem>

#include "neml2/drivers/ModelDriver.h"

namespace torch::nn
{
class ModuleDict;
}

namespace neml2
{
/**
 * @brief The driver for a transient initial-value problem.
 *
 */
class TransientDriver : public ModelDriver
{
public:
  static OptionSet expected_options();

  TransientDriver(const OptionSet & options);

  void setup() override;

  void diagnose() const override;

  bool run() override;

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
  /// Postprocess the output of the current time step.
  virtual void postprocess();
  /// Save the input of the current time step.
  virtual void store_input();
  // @}

  /// VariableName for the time
  const VariableName _time_name;
  /// The current time
  Scalar _time;
  /// The current step count
  Size _step_count = 0;
  /// Total number of steps
  const Size _nsteps;
  /// The input to the constitutive model
  ValueMap _in;

  /// The predictor used to set the initial guess
  const EnumSelection _predictor;

  /// Inputs from all time steps
  std::vector<ValueMap> _result_in;
  /// Outputs from all time steps
  std::vector<ValueMap> _result_out;

  /// Initial conditions
  ValueMap _ics;

  /// Driving forces (other than time)
  std::vector<VariableName> _driving_force_names;
  std::vector<Tensor> _driving_forces;

  /// Outputting
  ///@{
public:
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
  /// The destination file/path to save the results.
  virtual std::string save_as_path() const;

protected:
  /// Save the results into the destination file/path.
  virtual void output() const;

  /// The destination file name or file path
  std::string _save_as;

private:
  /// Output in torchscript format
  void output_pt(const std::filesystem::path & out) const;
  ///@}
};
} // namespace neml2
