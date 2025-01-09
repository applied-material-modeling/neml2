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

#include "neml2/models/solid_mechanics/ThermalEigenstrain.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"

namespace neml2
{
register_NEML2_object(ThermalEigenstrain);

OptionSet
ThermalEigenstrain::expected_options()
{
  OptionSet options = Eigenstrain::expected_options();
  options.doc() =
      "Define the (cummulative, as opposed to instantaneous) linear isotropic thermal eigenstrain, "
      "i.e. \\f$ \\boldsymbol{\\varepsilon}_T = \\alpha (T - T_0) \\boldsymbol{I} \\f$, where \\f$ "
      "\\alpha \\f$ is the coefficient of thermal expansion (CTE), \\f$ T \\f$ is the temperature, "
      "and \\f$ T_0 \\f$ is the reference (stress-free) temperature.";

  options.set_input("temperature") = VariableName(FORCES, "T");
  options.set("temperature").doc() = "Temperature";

  options.set_buffer<TensorName>("reference_temperature");
  options.set("reference_temperature").doc() = "Reference (stress-free) temperature";

  options.set_parameter<TensorName>("CTE");
  options.set("CTE").doc() = "Coefficient of thermal expansion";

  return options;
}

ThermalEigenstrain::ThermalEigenstrain(const OptionSet & options)
  : Eigenstrain(options),
    _T(declare_input_variable<Scalar>("temperature")),
    _T0(declare_buffer<Scalar>("T0", "reference_temperature")),
    _alpha(declare_parameter<Scalar>("alpha", "CTE", true))
{
}

void
ThermalEigenstrain::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _eg = _alpha * (_T - _T0) * SR2::identity(_T.options());

  if (dout_din)
  {
    if (_T.is_dependent())
      _eg.d(_T) = _alpha * SR2::identity(_T.options());

    if (const auto * const alpha = nl_param("alpha"))
      _eg.d(*alpha) = (_T - _T0) * SR2::identity(_T.options());
  }

  if (d2out_din2)
  {
    if (_T.is_dependent())
      if (const auto * const alpha = nl_param("alpha"))
      {
        _eg.d(_T, *alpha) = SR2::identity(_T.options());
        _eg.d(*alpha, _T) = SR2::identity(_T.options());
      }
  }
}
} // namespace neml2
