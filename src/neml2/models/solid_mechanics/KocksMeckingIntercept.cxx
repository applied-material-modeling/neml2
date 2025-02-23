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

#include "neml2/models/solid_mechanics/KocksMeckingIntercept.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(KocksMeckingIntercept);

OptionSet
KocksMeckingIntercept::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "The critical value of the normalized activation energy given by \\f$ g_0 "
                  "\\frac{C-B}{A} \\f$";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("A");
  options.set("A").doc() = "The Kocks-Mecking slope";

  options.set_parameter<TensorName<Scalar>>("B");
  options.set("B").doc() = "The Kocks-Mecking intercept";

  options.set_parameter<TensorName<Scalar>>("C");
  options.set("C").doc() = "The Kocks-Mecking horizontal value";

  options.set_output("intercept");
  options.set("intercept").doc() = "The intercept";

  return options;
}

KocksMeckingIntercept::KocksMeckingIntercept(const OptionSet & options)
  : Model(options),
    _A(declare_parameter<Scalar>("A", "A", true)),
    _B(declare_parameter<Scalar>("B", "B", true)),
    _C(declare_parameter<Scalar>("C", "C", true)),
    _b(declare_output_variable<Scalar>(VariableName(PARAMETERS, name())))
{
}

void
KocksMeckingIntercept::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _b = (_C - _B) / _A;

  if (dout_din)
  {
    if (const auto * const A = nl_param("A"))
      _b.d(*A) = -(_C - _B) / pow(_A, 2.0);

    if (const auto * const B = nl_param("B"))
      _b.d(*B) = -1.0 / _A;

    if (const auto * const C = nl_param("C"))
      _b.d(*C) = 1.0 / _A;
  }

  if (d2out_din2)
  {
    if (const auto * const A = nl_param("A"))
    {
      _b.d(*A, *A) = 2.0 * (_C - _B) / pow(_A, 3.0);
      if (const auto * const B = nl_param("B"))
        _b.d(*A, *B) = 1.0 / pow(_A, 2.0);
      if (const auto * const C = nl_param("C"))
        _b.d(*A, *C) = -1.0 / pow(_A, 2.0);
    }

    if (const auto * const B = nl_param("B"))
      if (const auto * const A = nl_param("A"))
        _b.d(*B, *A) = 1.0 / pow(_A, 2.0);

    if (const auto * const C = nl_param("C"))
      if (const auto * const A = nl_param("A"))
        _b.d(*C, *A) = -1.0 / pow(_A, 2.0);
  }
}
} // namespace neml2
