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

#include "neml2/models/finite_volume/FiniteVolumeUpwindedAdvectiveFlux.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"
#include "neml2/tensors/functions/sign.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{
register_NEML2_object(FiniteVolumeUpwindedAdvectiveFlux);

OptionSet
FiniteVolumeUpwindedAdvectiveFlux::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute upwinded advective fluxes at cell edges.";

  options.set_input("u") = VariableName(STATE, "u");
  options.set("u").doc() = "Cell-averaged field values.";

  options.set_input("v_edge") = VariableName(STATE, "v_edge");
  options.set("v_edge").doc() = "Cell-edge advection velocity values.";

  options.set_output("flux") = VariableName(STATE, "J_advection");
  options.set("flux").doc() = "Cell-edge advective fluxes.";

  return options;
}

FiniteVolumeUpwindedAdvectiveFlux::FiniteVolumeUpwindedAdvectiveFlux(const OptionSet & options)
  : Model(options),
    _u(declare_input_variable<Scalar>("u")),
    _v_edge(declare_input_variable<Scalar>("v_edge")),
    _J(declare_output_variable<Scalar>("flux"))
{
}

void
FiniteVolumeUpwindedAdvectiveFlux::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto N = _u.intmd_size(-1);
  const auto M = N - 1;
  const auto u_left = _u().intmd_slice(-1, indexing::Slice(0, N - 1));
  const auto u_right = _u().intmd_slice(-1, indexing::Slice(1, N));
  const auto v_vec = (_v_edge.intmd_dim() == 0 ? _v_edge().intmd_expand(M) : _v_edge());

  const auto v_abs = neml2::abs(v_vec);
  const auto v_plus = 0.5 * (v_vec + v_abs);
  const auto v_minus = 0.5 * (v_vec - v_abs);

  if (out)
    _J = v_plus * u_left + v_minus * u_right;

  if (dout_din)
  {
    // Selector matrices for adjacent pairs along the intermediate dimension.
    const auto u_map = imap_v<Scalar>(_u.options()).intmd_expand(N);
    const auto diag_u = intmd_diagonalize(u_map);
    const auto S_left_u = diag_u.intmd_slice(0, indexing::Slice(0, N - 1));
    const auto S_right_u = diag_u.intmd_slice(0, indexing::Slice(1, N));

    if (_u.is_dependent())
      _J.d(_u, 2, 1, 1) =
          v_plus.intmd_unsqueeze(1) * S_left_u + v_minus.intmd_unsqueeze(1) * S_right_u;

    if (_v_edge.is_dependent())
    {
      const auto s = neml2::sign(v_vec);
      const auto dvp = 0.5 * (1.0 + s);
      const auto dvm = 0.5 * (1.0 - s);
      const auto dJ_dvedge = dvp * u_left + dvm * u_right;

      if (_v_edge.intmd_dim() == 0)
        _J.d(_v_edge, 1, 1, 0) = dJ_dvedge;
      else
      {
        const auto v_map = imap_v<Scalar>(_v_edge.options()).intmd_expand(M);
        const auto diag_v = intmd_diagonalize(v_map);
        _J.d(_v_edge, 2, 1, 1) = dJ_dvedge.intmd_unsqueeze(1) * diag_v;
      }
    }
  }
}
} // namespace neml2
