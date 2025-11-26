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

#include "neml2/models/solid_mechanics/GTNYieldFunction.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/cosh.h"
#include "neml2/tensors/functions/sinh.h"

namespace neml2
{
register_NEML2_object(GTNYieldFunction);

OptionSet
GTNYieldFunction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Gurson-Tvergaard-Needleman yield function for poroplasticity. The yield function is defined "
      "as \\f$ f = \\left( \\frac{\\bar{\\sigma}}{\\sigma_y + k} \\right)^2 + 2 q_1 \\phi \\cosh "
      "\\left( \\frac{1}{2} q_2 \\frac{3\\sigma_h-\\sigma_s}{\\sigma_y + k} \\right) - \\left( q_3 "
      "\\phi^2 + 1 \\right) \\f$, where \\f$ \\bar{\\sigma} \\f$ is the von Mises stress, \\f$ "
      "\\sigma_y \\f$ is the yield stress, \\f$ k \\f$ is isotropic hardening, \\f$ \\phi \\f$ is "
      "the porosity, \\f$ \\sigma_h \\f$ is the hydrostatic stress, and \\f$ \\sigma_s \\f$ is the "
      "void growth back stress (sintering stress). \\f$ q_1 \\f$, \\f$ q_2 \\f$, and \\f$ q_3 \\f$ "
      "are parameters controlling the yield mechanisms.";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("yield_stress");
  options.set("yield_stress").doc() = "Yield stress";

  options.set_parameter<TensorName<Scalar>>("q1");
  options.set("q1").doc() =
      "Parameter controlling the balance/competition between plastic flow and void evolution.";

  options.set_parameter<TensorName<Scalar>>("q2");
  options.set("q2").doc() = "Void evolution rate";

  options.set_parameter<TensorName<Scalar>>("q3");
  options.set("q3").doc() = "Pore pressure";

  options.set_input("flow_invariant") = VariableName(STATE, "internal", "se");
  options.set("flow_invariant").doc() = "Effective stress driving plastic flow";

  options.set_input("poro_invariant") = VariableName(STATE, "internal", "sp");
  options.set("poro_invariant").doc() = "Effective stress driving porous flow";

  options.set_input("isotropic_hardening");
  options.set("isotropic_hardening").doc() = "Isotropic hardening";

  options.set_input("void_fraction") = VariableName(STATE, "internal", "f");
  options.set("void_fraction").doc() = "Void fraction (porosity)";

  options.set_output("yield_function") = VariableName(STATE, "internal", "fp");
  options.set("yield_function").doc() = "Yield function";

  return options;
}

GTNYieldFunction::GTNYieldFunction(const OptionSet & options)
  : Model(options),
    _f(declare_output_variable<Scalar>("yield_function")),
    _se(declare_input_variable<Scalar>("flow_invariant")),
    _sp(declare_input_variable<Scalar>("poro_invariant")),
    _phi(declare_input_variable<Scalar>("void_fraction")),
    _h(options.get<VariableName>("isotropic_hardening").empty()
           ? nullptr
           : &declare_input_variable<Scalar>("isotropic_hardening")),
    _s0(declare_parameter<Scalar>("sy", "yield_stress", /*allow_nonlinear=*/true)),
    _q1(declare_parameter<Scalar>("q1", "q1", /*allow_nonlinear=*/true)),
    _q2(declare_parameter<Scalar>("q2", "q2", /*allow_nonlinear=*/true)),
    _q3(declare_parameter<Scalar>("q3", "q3", /*allow_nonlinear=*/true))
{
}

void
GTNYieldFunction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // Flow stress (depending on whether isotropic hardening is provided)
  const auto sf = _h ? _s0 + (*_h) : _s0;

  if (out)
    _f = pow(_se / sf, 2.0) + 2 * _q1 * _phi * cosh(_q2 / 2.0 * _sp / sf) -
         (1.0 + _q3 * pow(_phi(), 2.0));

  if (dout_din)
  {
    if (_se.is_dependent())
      _f.d(_se) = 2.0 * _se / pow(sf, 2.0);

    if (_sp.is_dependent())
      _f.d(_sp) = _q1 * _phi * _q2 / sf * sinh(_q2 / 2.0 * _sp / sf);

    if (_phi.is_dependent())
      _f.d(_phi) = 2.0 * _q1 * cosh(_q2 / 2.0 * _sp / sf) - 2.0 * _q3 * _phi;

    if (_h)
      _f.d(*_h) = -2 * pow(_se(), 2.0) / pow(sf, 3.0) -
                  _q1 * _phi * _q2 * _sp / pow(sf, 2.0) * sinh(_q2 / 2.0 * _sp / sf);

    // Handle the case of variable coupling
    if (const auto * const sy = nl_param("sy"))
      _f.d(*sy) = -2 * pow(_se(), 2.0) / pow(sf, 3.0) -
                  _q1 * _phi * _q2 * _sp / pow(sf, 2.0) * sinh(_q2 / 2.0 * _sp / sf);

    if (const auto * const q1 = nl_param("q1"))
      _f.d(*q1) = 2.0 * _phi * cosh(_q2 / 2.0 * _sp / sf);

    if (const auto * const q2 = nl_param("q2"))
      _f.d(*q2) = _q1 * _phi * _sp / sf * sinh(_q2 / 2.0 * _sp / sf);

    if (const auto * const q3 = nl_param("q3"))
      _f.d(*q3) = -pow(_phi(), 2.0);
  }

  if (d2out_din2)
  {
    const auto * const sy = nl_param("sy");
    const auto * const q1 = nl_param("q1");
    const auto * const q2 = nl_param("q2");
    const auto * const q3 = nl_param("q3");

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // The GTN yield function can be expressed as
    //
    //      f(se, sp, phi, h; sy, q1, q2, q3)
    //
    //  - Arguments before the semicolon are variables
    //  - Arguments after the semicolon are (nonlinear) parameters
    //  - Derivatives w.r.t. the first three arguments se, sp, and phi are mandatory
    //  - Derivatives w.r.t. the rest of the arguments are optional
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // The second derivative is nothing but a big matrix. We will fill out the matrix row by row, in
    // the order of the arguments listed above.
    //
    // Rows will separated by big fences like this.
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // se: Flow invariant
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (_se.is_dependent())
    {
      _f.d2(_se, _se) = 2.0 / pow(sf, 2.0);

      if (_h)
        _f.d2(_se, *_h) = -4.0 * _se / pow(sf, 3.0);

      if (sy)
        _f.d2(_se, *sy) = -4.0 * _se / pow(sf, 3.0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // sp: Poro invariant
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (_sp.is_dependent())
    {
      _f.d2(_sp, _sp) =
          _phi * _q1 * pow(_q2, 2.0) / (2.0 * pow(sf, 2.0)) * cosh(_q2 / 2.0 * _sp / sf);

      if (_phi.is_dependent())
        _f.d2(_sp, _phi) = _q1 * _q2 * sinh(_q2 / 2.0 * _sp / sf) / sf;

      if (_h)
        _f.d2(_sp, *_h) =
            -_phi * _q1 * _q2 *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2 * pow(sf, 3.0));
      if (sy)
        _f.d2(_sp, *sy) =
            -_phi * _q1 * _q2 *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2 * pow(sf, 3.0));

      if (q1)
        _f.d2(_sp, *q1) = _phi * _q2 * sinh(_q2 / 2.0 * _sp / sf) / sf;

      if (q2)
        _f.d2(_sp, *q2) =
            _phi * _q1 *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2.0 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2.0 * pow(sf, 2.0));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // phi: Void fraction
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (_phi.is_dependent())
    {
      if (_sp.is_dependent())
        _f.d2(_phi, _sp) = _q1 * _q2 * sinh(_q2 / 2.0 * _sp / sf) / sf;

      _f.d2(_phi, _phi) = -2.0 * _q3;

      if (_h)
        _f.d2(_phi, *_h) = -_q1 * _q2 * _sp * sinh(_q2 / 2.0 * _sp / sf) / pow(sf, 2.0);

      if (sy)
        _f.d2(_phi, *sy) = -_q1 * _q2 * _sp * sinh(_q2 / 2.0 * _sp / sf) / pow(sf, 2.0);

      if (q1)
        _f.d2(_phi, *q1) = 2 * cosh(_q2 / 2.0 * _sp / sf);

      if (q2)
        _f.d2(_phi, *q2) = _q1 * _sp * sinh(_q2 / 2.0 * _sp / sf) / sf;

      if (q3)
        _f.d2(_phi, *q3) = -2.0 * _phi;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // h: (Optional) isotropic hardening
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (_h)
    {
      if (_se.is_dependent())
        _f.d2(*_h, _se) = -4.0 * _se / pow(sf, 3.0);

      if (_sp.is_dependent())
        _f.d2(*_h, _sp) =
            -_phi * _q1 * _q2 *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2.0 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2.0 * pow(sf, 3.0));

      if (_phi.is_dependent())
        _f.d2(*_h, _phi) = -_q1 * _q2 * _sp * sinh(_q2 / 2.0 * _sp / sf) / pow(sf, 2.0);

      _f.d2(*_h, *_h) = (12 * pow(_se(), 2.0) + _phi * _q1 * _q2 * _sp *
                                                    (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) +
                                                     4.0 * sf * sinh(_q2 / 2.0 * _sp / sf))) /
                        (2 * pow(sf, 4.0));

      if (sy)
        _f.d2(*_h, *sy) = (12 * pow(_se(), 2.0) + _phi * _q1 * _q2 * _sp *
                                                      (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) +
                                                       4.0 * sf * sinh(_q2 / 2.0 * _sp / sf))) /
                          (2 * pow(sf, 4.0));

      if (q1)
        _f.d2(*_h, *q1) = -_phi * _q2 * _sp * sinh(_q2 / 2.0 * _sp / sf) / pow(sf, 2.0);

      if (q2)
        _f.d2(*_h, *q2) =
            -_phi * _q1 * _sp *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2.0 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2 * pow(sf, 3.0));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // sy: (Optionally nonlinear) yield stress
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (sy)
    {
      if (_se.is_dependent())
        _f.d2(*sy, _se) = -4.0 * _se / pow(sf, 3.0);

      if (_phi.is_dependent())
        _f.d2(*sy, _phi) = -_q1 * _q2 * _sp * sinh(_q2 / 2.0 * _sp / sf) / pow(sf, 2.0);

      if (_h)
        _f.d2(*sy, *_h) = (12 * pow(_se(), 2.0) + _phi * _q1 * _q2 * _sp *
                                                      (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) +
                                                       4.0 * sf * sinh(_q2 / 2.0 * _sp / sf))) /
                          (2 * pow(sf, 4.0));

      _f.d2(*sy, *sy) = (12 * pow(_se(), 2.0) + _phi * _q1 * _q2 * _sp *
                                                    (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) +
                                                     4.0 * sf * sinh(_q2 / 2.0 * _sp / sf))) /
                        (2 * pow(sf, 4.0));

      if (q1)
        _f.d2(*sy, *q1) = -_phi * _q2 * _sp * sinh(_q2 / 2.0 * _sp / sf) / pow(sf, 2.0);

      if (q2)
        _f.d2(*sy, *q2) =
            -_phi * _q1 * _sp *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2.0 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2 * pow(sf, 3.0));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // q1: (Optionally nonlinear) GTN parameter q1
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (q1)
    {
      if (_sp.is_dependent())
        _f.d2(*q1, _sp) = _phi * _q2 * sinh(_q2 / 2.0 * _sp / sf) / sf;

      if (_phi.is_dependent())
        _f.d2(*q1, _phi) = 2.0 * cosh(_q2 / 2.0 * _sp / sf);

      if (_h)
        _f.d2(*q1, *_h) = -_phi * _q2 * _sp * sinh(_q2 / 2.0 * _sp / sf) / pow(sf, 2.0);

      if (sy)
        _f.d2(*q1, *sy) = -_phi * _q2 * _sp * sinh(_q2 / 2.0 * _sp / sf) / pow(sf, 2.0);

      if (q2)
        _f.d2(*q1, *q2) = _phi * _sp * sinh(_q2 / 2.0 * _sp / sf) / sf;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // q2: (Optionally nonlinear) GTN parameter q2
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (q2)
    {
      if (_sp.is_dependent())
        _f.d2(*q2, _sp) =
            _phi * _q1 *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2 * pow(sf, 2.0));

      if (_phi.is_dependent())
        _f.d2(*q2, _phi) = _q1 * _sp * sinh(_q2 / 2.0 * _sp / sf) / sf;

      if (_h)
        _f.d2(*q2, *_h) =
            -_phi * _q1 * _sp *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2 * pow(sf, 3.0));

      if (sy)
        _f.d2(*q2, *sy) =
            -_phi * _q1 * _sp *
            (_q2 * _sp * cosh(_q2 / 2.0 * _sp / sf) + 2 * sf * sinh(_q2 / 2.0 * _sp / sf)) /
            (2 * pow(sf, 3.0));

      if (q1)
        _f.d2(*q2, *q1) = _phi * _sp * sinh(_q2 / 2.0 * _sp / sf) / sf;

      _f.d2(*q2, *q2) =
          _phi * _q1 * pow(_sp(), 2.0) * cosh(_q2 / 2.0 * _sp / sf) / (2 * pow(sf, 2.0));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // q3: (Optionally nonlinear) GTN parameter q3
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (q3)
    {
      if (_phi.is_dependent())
        _f.d2(*q3, _phi) = -2.0 * _phi;
    }
  }
}
} // namespace neml2
