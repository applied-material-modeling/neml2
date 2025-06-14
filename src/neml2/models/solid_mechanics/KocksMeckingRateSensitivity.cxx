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

#include "neml2/models/solid_mechanics/KocksMeckingRateSensitivity.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(KocksMeckingRateSensitivity);

OptionSet
KocksMeckingRateSensitivity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculates the temperature-dependent rate sensitivity for a Perzyna-type model using the "
      "Kocks-Mecking model.  The value is \\f$ n = \\frac{\\mu b^3}{k T A} \\f$ with \\f$ \\mu "
      "\\f$ the shear modulus, \\f$ b \\f$ the Burgers vector, \\f$  k\\f$ the Boltzmann constant, "
      "\\f$ T \\f$ absolute temperature, and \\f$ A \\f$ the Kocks-Mecking slope parameter.";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("A");
  options.set("A").doc() = "The Kocks-Mecking slope parameter";
  options.set_parameter<TensorName<Scalar>>("shear_modulus");
  options.set("shear_modulus").doc() = "The shear modulus";

  options.set<double>("k");
  options.set("k").doc() = "Boltzmann constant";
  options.set<double>("b");
  options.set("b").doc() = "The Burgers vector";

  options.set_input("temperature") = VariableName(FORCES, "T");
  options.set("temperature").doc() = "Absolute temperature";

  return options;
}

KocksMeckingRateSensitivity::KocksMeckingRateSensitivity(const OptionSet & options)
  : Model(options),
    _A(declare_parameter<Scalar>("A", "A", /*allow_nonlinear=*/true)),
    _mu(declare_parameter<Scalar>("mu", "shear_modulus", /*allow_nonlinear=*/true)),
    _k(options.get<double>("k")),
    _b3(options.get<double>("b") * options.get<double>("b") * options.get<double>("b")),
    _T(declare_input_variable<Scalar>("temperature")),
    _m(declare_output_variable<Scalar>(VariableName(PARAMETERS, name())))
{
}

void
KocksMeckingRateSensitivity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _m = -_mu * _b3 / (_k * _T * _A);

  if (dout_din)
  {
    if (_T.is_dependent())
      _m.d(_T) = _b3 * _mu / (_A * _k * _T * _T);
    if (const auto * const mu = nl_param("mu"))
      _m.d(*mu) = -_b3 / (_A * _k * _T);
    if (const auto * const A = nl_param("A"))
      _m.d(*A) = _b3 * _mu / (_A * _A * _k * _T);
  }

  if (d2out_din2)
  {
    // T, T
    if (_T.is_dependent())
      _m.d(_T, _T) = -2.0 * _b3 * _mu / (_A * _k * _T * _T * _T);

    if (const auto * const A = nl_param("A"))
    {
      // A, A
      _m.d(*A, *A) = -2.0 * _b3 * _mu / (_A * _A * _A * _k * _T);
      // A, T and T, A
      if (_T.is_dependent())
      {
        auto AT = -_b3 * _mu / (_A * _A * _k * _T * _T);
        _m.d(*A, _T) = AT;
        _m.d(_T, *A) = AT;
      }
    }

    if (const auto * const mu = nl_param("mu"))
    {
      // mu, T and T, mu
      if (_T.is_dependent())
      {
        auto MT = _b3 / (_A * _k * _T * _T);
        _m.d(*mu, _T) = MT;
        _m.d(_T, *mu) = MT;
      }

      if (const auto * const A = nl_param("A"))
      {
        // mu, A and A, mu
        auto MA = _b3 / (_A * _A * _k * _T);
        _m.d(*mu, *A) = MA;
        _m.d(*A, *mu) = MA;
      }
    }
  }
}
} // namespace neml2
