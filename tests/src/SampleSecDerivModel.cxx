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

#include "SampleSecDerivModel.h"

using namespace neml2;

register_NEML2_object(SampleSecDerivModel);
register_NEML2_object(ADSampleSecDerivModel);

template <bool is_ad>
SampleSecDerivModelTempl<is_ad>::SampleSecDerivModelTempl(const ParameterSet & params)
  : SampleSecDerivModelBase<is_ad>(params)
{
  this->input().template add<LabeledAxis>("state");
  this->input().subaxis("state").template add<Scalar>("x1");
  this->input().subaxis("state").template add<Scalar>("x2");

  this->output().template add<LabeledAxis>("state");
  this->output().subaxis("state").template add<Scalar>("y");

  this->setup();
}

template <bool is_ad>
void
SampleSecDerivModelTempl<is_ad>::set_value(LabeledVector in,
                                           LabeledVector out,
                                           LabeledMatrix * dout_din) const
{
  // Grab the inputs
  auto x1 = in.slice("state").get<Scalar>("x1");
  auto x2 = in.slice("state").get<Scalar>("x2");

  // y = x1^3 + x2^4
  auto y = x1 * x1 * x1 + x2 * x2 * x2 * x2;

  // Set the output
  out.slice("state").set(y, "y");

  if constexpr (!is_ad)
    if (dout_din)
    {
      auto dy_dx1 = 3 * x1 * x1;
      auto dy_dx2 = 4 * x2 * x2 * x2;

      dout_din->block("state", "state").set(dy_dx1, "y", "x1");
      dout_din->block("state", "state").set(dy_dx2, "y", "x2");
    }
}

template <bool is_ad>
void
SampleSecDerivModelTempl<is_ad>::set_dvalue(LabeledVector in,
                                            LabeledMatrix dout_din,
                                            LabeledTensor<1, 3> * d2out_din2) const
{
  // Grab the inputs
  auto x1 = in.slice("state").get<Scalar>("x1");
  auto x2 = in.slice("state").get<Scalar>("x2");

  // y = x1^3 + x2^4
  auto dy_dx1 = 3 * x1 * x1;
  auto dy_dx2 = 4 * x2 * x2 * x2;

  // Set the output
  dout_din.block("state", "state").set(dy_dx1, "y", "x1");
  dout_din.block("state", "state").set(dy_dx2, "y", "x2");

  if constexpr (!is_ad)
    if (d2out_din2)
    {
      auto d2y_dx12 = 6 * x1;
      auto d2y_dx22 = 12 * x2 * x2;

      d2out_din2->block("state", "state", "state").set(d2y_dx12, "y", "x1", "x1");
      d2out_din2->block("state", "state", "state").set(d2y_dx22, "y", "x2", "x2");
    }
}

template class SampleSecDerivModelTempl<true>;
template class SampleSecDerivModelTempl<false>;
