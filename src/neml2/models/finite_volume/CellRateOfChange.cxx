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

#include "neml2/models/finite_volume/CellRateOfChange.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diff.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{
register_NEML2_object(CellRateOfChange);

OptionSet
CellRateOfChange::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute cell rate of change from flux divergence and reaction.";

  options.set_input("flux") = VariableName(STATE, "J");
  options.set("flux").doc() = "Cell-edge total fluxes.";

  options.set_parameter<TensorName<Scalar>>("cell_size");
  options.set("cell_size").doc() = "Cell sizes.";

  options.set_input("reaction") = VariableName(STATE, "R");
  options.set("reaction").doc() = "Cell reaction rates.";

  options.set_output("rate") = VariableName(STATE, "u_rate");
  options.set("rate").doc() = "Cell rate of change.";

  return options;
}

CellRateOfChange::CellRateOfChange(const OptionSet & options)
  : Model(options),
    _flux(declare_input_variable<Scalar>("flux")),
    _dx(declare_parameter<Scalar>("cell_size", "cell_size", true)),
    _R(declare_input_variable<Scalar>("reaction")),
    _u_dot(declare_output_variable<Scalar>("rate"))
{
}

void
CellRateOfChange::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto dJ = intmd_diff(_flux(), 1, -1);
  const auto N = _R.intmd_size(-1);
  const auto dx_vec = (_dx.intmd_dim() == 0 ? _dx.intmd_expand(N) : _dx);
  const auto inv_dx = 1.0 / dx_vec;

  if (out)
    _u_dot = _R - dJ * inv_dx;

  if (dout_din)
  {
    const auto I = intmd_diagonalize(imap_v<Scalar>(_R.options()).intmd_expand(N));

    if (_R.is_dependent())
      _u_dot.d(_R, 2, 1, 1) = I;

    if (const auto * const dx = nl_param("cell_size"))
    {
      const auto dJ_over_dx2 = dJ * inv_dx * inv_dx;
      if (dx->intmd_dim() == 0)
        _u_dot.d(*dx, 1, 1, 0) = dJ_over_dx2;
      else
        _u_dot.d(*dx, 2, 1, 1) = dJ_over_dx2.intmd_unsqueeze(1) * I;
    }

    if (_flux.is_dependent())
    {
      // Selector matrices for adjacent edges: S_left picks J_{i-1/2}, S_right picks J_{i+1/2}
      const auto m = _flux.intmd_size(-1);
      const auto Jmap = imap_v<Scalar>(_flux.options()).intmd_expand(m);
      const auto diagJ = intmd_diagonalize(Jmap);
      const auto S_left = diagJ.intmd_slice(0, indexing::Slice(0, m - 1));
      const auto S_right = diagJ.intmd_slice(0, indexing::Slice(1, m));
      const auto dJ_dflux = S_right - S_left;

      _u_dot.d(_flux, 2, 1, 1) = (-inv_dx).intmd_unsqueeze(1) * dJ_dflux;
    }
  }
}
} // namespace neml2
