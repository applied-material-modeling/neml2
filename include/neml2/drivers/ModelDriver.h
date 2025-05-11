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
#include "neml2/models/map_types.h"
#include "neml2/tensors/tensors.h"

#ifdef NEML2_HAS_DISPATCHER
#include "neml2/dispatchers/WorkScheduler.h"
#include "neml2/dispatchers/WorkDispatcher.h"
#endif

namespace neml2
{
class Model;

/// Callback to dump a ton of information on model exectuion
void details_callback(const Model &,
                      const std::map<VariableName, std::unique_ptr<VariableBase>> &,
                      const std::map<VariableName, std::unique_ptr<VariableBase>> &);

/**
 * @brief A general-purpose driver that does *something* with a model
 *
 * Derived classes define that "something".
 */
class ModelDriver : public Driver
{
public:
  static OptionSet expected_options();

  ModelDriver(const OptionSet & options);

  void setup() override;

  void diagnose() const override;

  const Model & model() const { return _model; }

protected:
  /// The model which the driver uses to perform constitutive updates.
  Model & _model;
  /// The device on which to evaluate the model
  const Device _device;

  /// Set to true to list all the model parameters at the beginning
  const bool _show_params;
  /// Set to true to show model's input axis at the beginning
  const bool _show_input;
  /// Set to true to show model's output axis at the beginning
  const bool _show_output;
  /// Set to output a ton of information on the model execution
  const bool _log_details;

#ifdef NEML2_HAS_DISPATCHER
  /// The work scheduler to use
  std::shared_ptr<WorkScheduler> _scheduler;
  /// Work dispatcher
  using DispatcherType = WorkDispatcher<ValueMap, ValueMap, ValueMap, ValueMap, ValueMap>;
  std::unique_ptr<DispatcherType> _dispatcher;
  /// Whether to dispatch work asynchronously
  const bool _async_dispatch;
#endif
};
} // namespace neml2
