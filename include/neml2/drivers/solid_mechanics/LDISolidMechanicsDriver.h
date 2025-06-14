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

#include "neml2/drivers/solid_mechanics/SolidMechanicsDriver.h"

#include "neml2/tensors/WR2.h"

namespace neml2
{
/// Large deformation incremental solid mechanics driver
class LDISolidMechanicsDriver : public SolidMechanicsDriver
{
public:
  static OptionSet expected_options();

  LDISolidMechanicsDriver(const OptionSet & options);

  void setup() override;

  void diagnose() const override;

protected:
  void init_strain_control(const OptionSet & options) override;
  void init_stress_control(const OptionSet & options) override;
  virtual void init_vorticity_control(const OptionSet & options);

  void update_forces() override;

  void apply_predictor() override;

  ///@{
  /// Whether vorticity is prescribed
  const bool _vorticity_prescribed;
  /// The name of the total vorticity
  VariableName _vorticity_name;
  /// The value of the (total) vorticity
  WR2 _vorticity;
  ///@}

  ///@{
  /// Whether to perform the CP warmup
  const bool _cp_warmup;
  /// Scale value for initial cp warmup
  const double _cp_warmup_elastic_scale;
  /// Name of the elastic strain variable for the CP warmup
  const VariableName _cp_warmup_elastic_strain;
  ///@}
};
}
