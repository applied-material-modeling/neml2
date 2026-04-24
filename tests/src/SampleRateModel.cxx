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

#include "SampleRateModel.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/tr.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
using SampleRateModel = SampleRateModelTmpl<false>;
using ADSampleRateModel = SampleRateModelTmpl<true>;

register_NEML2_object(SampleRateModel);
register_NEML2_object(ADSampleRateModel);

template <bool AD>
OptionSet
SampleRateModelTmpl<AD>::expected_options()
{
  auto options = Model::expected_options();
  options.doc() = "A sample model that computes rates for a set of variables.";

  options.add_input("foo", "A variable of type Scalar");
  options.add_input("bar", "A variable of type Scalar");
  options.add_input("baz", "A variable of type SR2");
  options.add_input("temperature", "Temperature variable");

  return options;
}

template <bool AD>
SampleRateModelTmpl<AD>::SampleRateModelTmpl(const OptionSet & options)
  : Model(options),
    foo(declare_input_variable<Scalar>("foo")),
    bar(declare_input_variable<Scalar>("bar")),
    baz(declare_input_variable<SR2>("baz")),
    T(declare_input_variable<Scalar>("temperature")),
    foo_dot(declare_output_variable<Scalar>(rate_name(foo.name()))),
    bar_dot(declare_output_variable<Scalar>(rate_name(bar.name()))),
    baz_dot(declare_output_variable<SR2>(rate_name(baz.name()))),
    _a(declare_parameter<Scalar>("a", Scalar::full(-0.01))),
    _b(declare_parameter<Scalar>("b", Scalar::full(-0.5))),
    _c(declare_parameter<Scalar>("c", Scalar::full(-0.9)))
{
}

template <>
void
SampleRateModelTmpl<false>::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
  {
    foo_dot = (foo * foo + bar) * T + neml2::tr(baz());
    bar_dot = _a * bar + _b * foo + _c * T + neml2::tr(baz());
    baz_dot = (foo + bar) * baz * (T - 3);
  }

  if (dout_din)
  {
    auto I = SR2::identity(foo.options());

    foo_dot.d(foo) = 2 * foo * T;
    foo_dot.d(bar) = T();
    foo_dot.d(baz) = I;

    bar_dot.d(foo) = _b;
    bar_dot.d(bar) = _a;
    bar_dot.d(baz) = I;

    baz_dot.d(foo) = baz * (T - 3);
    baz_dot.d(bar) = baz * (T - 3);
    baz_dot.d(baz) = (foo + bar) * (T - 3) * imap_v<SR2>(foo.options());

    foo_dot.d(T) = foo * foo + bar;
    bar_dot.d(T) = _c;
    baz_dot.d(T) = (foo + bar) * baz;
  }
}

template <>
void
SampleRateModelTmpl<true>::set_value(bool out, bool /*dout_din*/, bool /*d2out_din2*/)
{
  if (out)
  {
    foo_dot = (foo * foo + bar) * T + neml2::tr(baz());
    bar_dot = _a * bar + _b * foo + _c * T + neml2::tr(baz());
    baz_dot = (foo + bar) * baz * (T - 3);
  }
}

template <bool AD>
void
SampleRateModelTmpl<AD>::request_AD()
{
  if constexpr (AD)
  {
    std::vector<const VariableBase *> inputs = {&foo, &bar, &baz, &T};

    // First derivatives
    foo_dot.request_AD(inputs);
    bar_dot.request_AD(inputs);
    baz_dot.request_AD(inputs);

    // Second derivatives
    foo_dot.request_AD(inputs, inputs);
    bar_dot.request_AD(inputs, inputs);
    baz_dot.request_AD(inputs, inputs);
  }
}
}
