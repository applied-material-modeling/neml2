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

#include "neml2/models/finite_volume/SemiInfiniteCoordinateTransform.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(SemiInfiniteCoordinateTransform);

OptionSet
SemiInfiniteCoordinateTransform::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Transform a semi-infinite coordinate to x / (x + s).";

  options.set_input("coordinate");
  options.set("coordinate").doc() = "Input coordinate.";

  options.set_parameter<TensorName<Scalar>>("shift");
  options.set("shift").doc() = "Shift parameter.";

  options.set_output("transformed_coordinate") = VariableName(STATE, "x_hat");
  options.set("transformed_coordinate").doc() = "Transformed coordinate.";

  return options;
}

SemiInfiniteCoordinateTransform::SemiInfiniteCoordinateTransform(const OptionSet & options)
  : Model(options),
    _x(declare_input_variable<Scalar>("coordinate")),
    _s(declare_parameter<Scalar>("s", "shift", true)),
    _x_hat(declare_output_variable<Scalar>("transformed_coordinate"))
{
}

void
SemiInfiniteCoordinateTransform::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto x = _x();
  const auto s = _s;
  const auto denom = x + s;
  const auto inv = 1.0 / denom;
  const auto x_hat = x * inv;

  if (out)
    _x_hat = x_hat;

  if (dout_din)
  {
    if (_x.is_dependent())
    {
      const auto dx = s * inv * inv;
      if (_x.intmd_dim() == 0)
        _x_hat.d(_x) = dx;
      else
      {
        const auto nbin = _x.intmd_size(-1);
        const auto x_map = imap_v<Scalar>(_x.options()).intmd_expand(nbin);
        const auto diag_x = intmd_diagonalize(x_map);
        _x_hat.d(_x, 2, 1, 1) = dx.intmd_unsqueeze(1) * diag_x;
      }
    }

    if (const auto * const s_param = nl_param("s"))
    {
      const auto ds = -x * inv * inv;
      if (s_param->intmd_dim() == 0)
        _x_hat.d(*s_param) = ds;
      else
      {
        const auto nbin = s_param->intmd_size(-1);
        const auto s_map = imap_v<Scalar>(_s.options()).intmd_expand(nbin);
        const auto diag_s = intmd_diagonalize(s_map);
        _x_hat.d(*s_param, 2, 1, 1) = ds.intmd_unsqueeze(1) * diag_s;
      }
    }
  }
}
} // namespace neml2
