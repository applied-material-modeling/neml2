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

namespace neml2
{
register_NEML2_object(KocksMeckingRateSensitivity);

OptionSet
KocksMeckingRateSensitivity::expected_options()
{
  OptionSet options = NonlinearParameter<Scalar>::expected_options();
  options.doc() =
      "Calculates the temperature-dependent rate sensitivity for a Perzyna-type model using the "
      "Kocks-Mecking model.  The value is \\f$ n = \\frac{\\mu b^3}{k T A} \\f$ with \\f$ \\mu "
      "\\f$ the shear modulus, \\f$ b \\f$ the Burgers vector, \\f$  k\\f$ the Boltzmann constant, "
      "\\f$ T \\f$ absolute temperature, and \\f$ A \\f$ the Kocks-Mecking slope parameter.";

  options.set_parameter<CrossRef<Scalar>>("A");
  options.set("A").doc() = "The Kocks-Mecking slope parameter";
  options.set_parameter<CrossRef<Scalar>>("shear_modulus");
  options.set("shear_modulus").doc() = "The shear modulus";

  options.set<Real>("k");
  options.set("k").doc() = "Boltzmann constant";
  options.set<Real>("b");
  options.set("b").doc() = "The Burgers vector";

  options.set_input("temperature") = VariableName(FORCES, "T");
  options.set("temperature").doc() = "Absolute temperature";
  return options;
}

KocksMeckingRateSensitivity::KocksMeckingRateSensitivity(const OptionSet & options)
  : NonlinearParameter<Scalar>(options),
    _A(declare_parameter<Scalar>("A", "A", /*allow_nonlinear=*/true)),
    _mu(declare_parameter<Scalar>("mu", "shear_modulus", /*allow_nonlinear=*/true)),
    _k(options.get<Real>("k")),
    _b3(options.get<Real>("b") * options.get<Real>("b") * options.get<Real>("b")),
    _T(declare_input_variable<Scalar>("temperature"))
{
}

void
KocksMeckingRateSensitivity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _p = -_mu * _b3 / (_k * _T * _A);

  if (dout_din)
  {
    if (_T.is_dependent())
      _p.d(_T) = _b3 * _mu / (_A * _k * _T * _T);
    if (const auto * const mu = nl_param("mu"))
      _p.d(*mu) = -_b3 / (_A * _k * _T);
    if (const auto * const A = nl_param("A"))
      _p.d(*A) = _b3 * _mu / (_A * _A * _k * _T);
  }

  if (d2out_din2)
  {
    // T, T
    if (_T.is_dependent())
      _p.d(_T, _T) = -2.0 * _b3 * _mu / (_A * _k * _T * _T * _T);

    if (const auto * const A = nl_param("A"))
    {
      // A, A
      _p.d(*A, *A) = -2.0 * _b3 * _mu / (_A * _A * _A * _k * _T);
      // A, T and T, A
      if (_T.is_dependent())
      {
        auto AT = -_b3 * _mu / (_A * _A * _k * _T * _T);
        _p.d(*A, _T) = AT;
        _p.d(_T, *A) = AT;
      }
    }

    if (const auto * const mu = nl_param("mu"))
    {
      // mu, T and T, mu
      if (_T.is_dependent())
      {
        auto MT = _b3 / (_A * _k * _T * _T);
        _p.d(*mu, _T) = MT;
        _p.d(_T, *mu) = MT;
      }

      if (const auto * const A = nl_param("A"))
      {
        // mu, A and A, mu
        auto MA = _b3 / (_A * _A * _k * _T);
        _p.d(*mu, *A) = MA;
        _p.d(*A, *mu) = MA;
      }
    }
  }
}
} // namespace neml2
