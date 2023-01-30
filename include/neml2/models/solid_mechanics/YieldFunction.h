// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/models/SecDerivModel.h"
#include "neml2/models/solid_mechanics/StressMeasure.h"
#include "neml2/tensors/LabeledAxis.h"

namespace neml2
{
class YieldFunction : public SecDerivModel
{
public:
  /// Calculate yield function knowing the corresponding hardening model
  YieldFunction(const std::string & name,
                const std::shared_ptr<StressMeasure> & sm,
                Scalar s0,
                bool with_isotropic_hardening,
                bool with_kinematic_hardening);

  const LabeledAxisAccessor mandel_stress;
  const LabeledAxisAccessor yield_function;
  const LabeledAxisAccessor isotropic_hardening;
  const LabeledAxisAccessor kinematic_hardening;
  /// Stress measure
  const StressMeasure & stress_measure;

protected:
  /// The value of the yield function
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  /// The derivative of the yield function w.r.t. hardening variables
  virtual void set_dvalue(LabeledVector in,
                          LabeledMatrix dout_din,
                          LabeledTensor<1, 3> * d2out_din2 = nullptr) const;

private:
  LabeledVector make_stress_measure_input(LabeledVector in) const;

protected:
  /// Yield stress
  Scalar _s0;

  /// @{
  /// Whether we include isotropic and/or kinematic hardening
  const bool _with_isotropic_hardening;
  const bool _with_kinematic_hardening;
  /// @}
};
} // namespace neml2
