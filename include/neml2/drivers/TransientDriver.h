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
   * @return torch::nn::ModuleDict The results (input and output) from all time steps. Keys of the
   * dict are "input" and "output". Each buffer in the submodules correspond to a variable.
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
  /// Save the output of the current time step.
  virtual void store_output();
  // @}

  /// Save the results into the destination file/path.
  virtual void output() const;

  /// The model which the driver uses to perform constitutive updates.
  Model & _model;
  /// The device on which to evaluate the model
  const torch::Device _device;

  /// The current time
  Scalar _time;
  /// The current step count
  Size _step_count;
  /// VariableName for the time
  VariableName _time_name;
  /// Total number of steps
  Size _nsteps;
  /// The batch size
  Size _nbatch;
  /// The input to the constitutive model
  LabeledVector _in;
  /// The output of the constitutive model
  LabeledVector _out;

  /// The predictor used to set the initial guess
  std::string _predictor;
  /// The destination file name or file path
  std::string _save_as;
  /// Set to true to list all the model parameters at the beginning
  const bool _show_params;
  /// Set to true to show model's input axis at the beginning
  const bool _show_input;
  /// Set to true to show model's output axis at the beginning
  const bool _show_output;

  /// Inputs from all time steps
  LabeledVector _result_in;
  /// Outputs from all time steps
  LabeledVector _result_out;

  /// Names for scalar initial conditions
  std::vector<VariableName> _ic_scalar_names;
  /// Values for the scalar initial conditions
  std::vector<CrossRef<Scalar>> _ic_scalar_values;
  /// Names for the Rot initial conditions
  std::vector<VariableName> _ic_rot_names;
  /// Values for the Rot initial conditions
  std::vector<CrossRef<Rot>> _ic_rot_values;
  /// Names for the SR2 initial conditions
  std::vector<VariableName> _ic_sr2_names;
  /// Values for the SR2 initial conditions
  std::vector<CrossRef<SR2>> _ic_sr2_values;

  /// Scale value for initial cp predictor
  Real _cp_elastic_scale;

private:
  void output_pt(const std::filesystem::path & out) const;

  template <typename T>
  void set_IC(const std::vector<VariableName> & ic_names,
              const std::vector<CrossRef<T>> & ic_values)
  {
    for (size_t i = 0; i < ic_names.size(); i++)
      _out.base_index_put_(ic_names[i], T(ic_values[i]));
  }
};
} // namespace neml2
