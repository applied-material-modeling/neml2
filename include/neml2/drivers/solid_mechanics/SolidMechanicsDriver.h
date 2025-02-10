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

#include "neml2/drivers/TransientDriver.h"

namespace neml2
{
/**
 * @brief The transient driver specialized for solid mechanics problems.
 *
 * Derived classes should implement the `SolidMechanicsDriver::update_forces` method to update the
 * driving forces based on the assumptions on kinematics, i.e., small vs large deformation,
 * incremental vs total etc.
 *
 */
class SolidMechanicsDriver : public TransientDriver
{
public:
  static OptionSet expected_options();

  SolidMechanicsDriver(const OptionSet & options);

  void setup() override;

  void diagnose() const override;

protected:
  void update_forces() override;

  ///@{
  /// Control-specific initialization
  virtual void init_strain_control(const OptionSet &) = 0;
  virtual void init_stress_control(const OptionSet &) = 0;
  virtual void init_mixed_control(const OptionSet &);
  virtual void init_temperature_control(const OptionSet &);
  ///@}

  /**
   * @brief The control method to drive the constitutive update.
   *
   * STRAIN: Use strain control to drive the update.
   * STRESS: Use stress control to drive the update.
   * MIXED: Use mixed stress/strain control with signal provided by _control_name
   */
  const EnumSelection _control;

  ///@{
  /// The name of the driving force
  VariableName _driving_force_name;
  /// The value of the driving force
  SR2 _driving_force;
  ///@}

  ///@{
  /// Name of the control signal for mixed stress/strain control
  VariableName _mixed_control_name;
  /// Actual control signal, when used for control == "MIXED"
  SR2 _mixed_control;
  ///@}

  ///@{
  /// Whether temperature is prescribed
  const bool _temperature_prescribed;
  /// Name of the temperature variable
  VariableName _temperature_name;
  /// Temperature
  Scalar _temperature;
  ///@}
};
}
