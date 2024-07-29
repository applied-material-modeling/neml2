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

#include "neml2/models/solid_mechanics/KocksMeckingFlowViscosity.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(KocksMeckingFlowViscosity);

OptionSet
KocksMeckingFlowViscosity::expected_options()
{
  OptionSet options = NonlinearParameter<Scalar>::expected_options();
  options.doc() =
      "Calculates the temperature-dependent flow viscosity for a Perzyna-type model using the "
      "Kocks-Mecking model.  The value is \\f$ \\eta = \\exp{B} \\mu "
      "\\dot{\\varepsilon}_0^\\frac{-k T A}{\\mu b^3} \\f$ with \\f$ \\mu "
      "\\f$ the shear modulus, \\f$ \\dot{\\varepsilon}_0 \\f$ a reference strain rate,  \\f$ b "
      "\\f$ the Burgers vector, "
      "\\f$  k\\f$ the Boltzmann constant, "
      "\\f$ T \\f$ absolute temperature, \\f$ A \\f$ the Kocks-Mecking slope parameter, and \\f$ B "
      "\\f$ the Kocks-Mecking intercept parameter.";

  options.set_parameter<CrossRef<Scalar>>("A");
  options.set("A").doc() = "The Kocks-Mecking slope parameter";
  options.set_parameter<CrossRef<Scalar>>("B");
  options.set("B").doc() = "The Kocks-Mecking intercept parameter";
  options.set_parameter<CrossRef<Scalar>>("shear_modulus");
  options.set("shear_modulus").doc() = "The shear modulus";

  options.set<Real>("eps0");
  options.set("eps0").doc() = "The reference strain rate";

  options.set<Real>("k");
  options.set("k").doc() = "Boltzmann constant";
  options.set<Real>("b");
  options.set("b").doc() = "The Burgers vector";

  options.set_input("temperature") = VariableName("forces", "T");
  options.set("temperature").doc() = "Absolute temperature";

  return options;
}

KocksMeckingFlowViscosity::KocksMeckingFlowViscosity(const OptionSet & options)
  : NonlinearParameter<Scalar>(options),
    _A(declare_parameter<Scalar>("A", "A", /*allow_nonlinear=*/true)),
    _B(declare_parameter<Scalar>("B", "B", /*allow_nonlinear=*/true)),
    _mu(declare_parameter<Scalar>("mu", "shear_modulus", /*allow_nonlinear=*/true)),
    _eps0(options.get<Real>("eps0")),
    _k(options.get<Real>("k")),
    _b3(options.get<Real>("b") * options.get<Real>("b") * options.get<Real>("b")),
    _T(declare_input_variable<Scalar>("temperature"))
{
}

void
KocksMeckingFlowViscosity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto post = math::pow(_eps0, _k * _T * _A / (_mu * _b3));

  if (out)
    _p = math::exp(_B) * _mu * post;

  if (dout_din)
  {
    if (_T.is_dependent())
      _p.d(_T) = _A * math::exp(_B) * _k * std::log(_eps0) * post / _b3;

    if (const auto * const A = nl_param("A"))
      _p.d(*A) = math::exp(_B) * _k * _T * std::log(_eps0) * post / _b3;

    if (const auto * const B = nl_param("B"))
      _p.d(*B) = math::exp(_B) * _mu * post;

    if (const auto * const mu = nl_param("mu"))
      _p.d(*mu) = math::exp(_B) * post * (1.0 - _A * _k * _T * std::log(_eps0) / (_b3 * _mu));
  }

  if (d2out_din2)
  {
    // T
    if (_T.is_dependent())
    {
      _p.d(_T, _T) = math::pow(_A * _k * std::log(_eps0) / _b3, 2.0) * math::exp(_B) * post / _mu;

      if (const auto * const A = nl_param("A"))
        _p.d(_T, *A) = math::exp(_B) * _k * std::log(_eps0) * post / _b3 *
                       (std::log(_eps0) * _A * _k * _T / (_b3 * _mu) + 1.0);

      if (const auto * const B = nl_param("B"))
        _p.d(_T, *B) = _A * math::exp(_B) * _k * std::log(_eps0) * post / _b3;

      if (const auto * const mu = nl_param("mu"))
        _p.d(_T, *mu) =
            -math::pow(_A * _k * std::log(_eps0) / (_b3 * _mu), 2.0) * math::exp(_B) * _T * post;
    }

    // A
    if (const auto * const A = nl_param("A"))
    {
      if (_T.is_dependent())
        _p.d(*A, _T) = math::exp(_B) * _k * std::log(_eps0) * post / _b3 *
                       (std::log(_eps0) * _A * _k * _T / (_b3 * _mu) + 1.0);

      _p.d(*A, *A) = math::exp(_B) * math::pow(_k * _T * std::log(_eps0) / _b3, 2.0) * post / _mu;

      if (const auto * const B = nl_param("B"))
        _p.d(*A, *B) = math::exp(_B) * _k * _T * std::log(_eps0) * post / _b3;

      if (const auto * const mu = nl_param("mu"))
        _p.d(*A, *mu) =
            -_A * math::exp(_B) * math::pow(_k * _T * std::log(_eps0) / (_b3 * _mu), 2.0) * post;
    }

    // B
    if (const auto * const B = nl_param("B"))
    {
      if (_T.is_dependent())
        _p.d(*B, _T) = _A * math::exp(_B) * _k * std::log(_eps0) * post / _b3;

      if (const auto * const A = nl_param("A"))
        _p.d(*B, *A) = math::exp(_B) * _k * _T * std::log(_eps0) * post / _b3;

      _p.d(*B, *B) = math::exp(_B) * _mu * post;

      if (const auto * const mu = nl_param("mu"))
        _p.d(*B, *mu) =
            math::exp(_B) * post * (_b3 * _mu - _A * _k * _T * std::log(_eps0)) / (_b3 * _mu);
    }

    // mu
    if (const auto * const mu = nl_param("mu"))
    {
      if (_T.is_dependent())
        _p.d(*mu, _T) =
            -math::exp(_B) * math::pow(_A * _k * std::log(_eps0) / (_b3 * _mu), 2.0) * _T * post;

      if (const auto * const A = nl_param("A"))
        _p.d(*mu, *A) =
            -_A * math::exp(_B) * math::pow(_k * _T * std::log(_eps0) / (_b3 * _mu), 2.0) * post;

      if (const auto * const B = nl_param("B"))
        _p.d(*mu, *B) =
            math::exp(_B) * post * (_b3 * _mu - _A * _k * _T * std::log(_eps0)) / (_b3 * _mu);

      _p.d(*mu, *mu) = -math::pow(_A * _k * _T * std::log(_eps0) / (_b3 * _mu), 2.0) *
                       math::exp(_B) * post / _mu;
    }
  }
}
} // namespace neml2
