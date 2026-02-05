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

#include "neml2/models/transport/DiffusiveFlux.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{
register_NEML2_object(DiffusiveFlux);

OptionSet
DiffusiveFlux::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute diffusive fluxes at cell edges using first-order reconstruction.";

  options.set_input("u") = VariableName(STATE, "u");
  options.set("u").doc() = "Cell-averaged field values.";

  options.set_input("D_edge") = VariableName(STATE, "D_edge");
  options.set("D_edge").doc() = "Cell-edge diffusivity values.";

  options.set_output("flux") = VariableName(STATE, "J_diffusion");
  options.set("flux").doc() = "Cell-edge diffusive fluxes.";

  return options;
}

DiffusiveFlux::DiffusiveFlux(const OptionSet & options)
  : Model(options),
    _u(declare_input_variable<Scalar>("u")),
    _D_edge(declare_input_variable<Scalar>("D_edge")),
    _J(declare_output_variable<Scalar>("flux"))
{
}

void
DiffusiveFlux::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto N = _u.intmd_size(-1);
  const auto M = N - 1;
  const auto u_left = _u().intmd_slice(-1, indexing::Slice(0, N - 1));
  const auto u_right = _u().intmd_slice(-1, indexing::Slice(1, N));
  const auto D_vec = (_D_edge.intmd_dim() == 0 ? _D_edge().intmd_expand(M) : _D_edge());

  const auto du = u_right - u_left;

  if (out)
    _J = -D_vec * du;

  if (dout_din)
  {
    // Build selector matrices for adjacent pairs. S_left picks i, S_right picks i+1.
    const auto u_map = imap_v<Scalar>(_u.options()).intmd_expand(N);
    const auto diag_u = intmd_diagonalize(u_map);
    const auto S_left = diag_u.intmd_slice(0, indexing::Slice(0, N - 1));
    const auto S_right = diag_u.intmd_slice(0, indexing::Slice(1, N));

    // dJ/du = -D_edge * (S_right - S_left)
    if (_u.is_dependent())
      _J.d(_u, 2, 1, 1) = (-D_vec).intmd_unsqueeze(1) * (S_right - S_left);

    // dJ/dD_edge = -du * I
    if (_D_edge.is_dependent())
    {
      if (_D_edge.intmd_dim() == 0)
        _J.d(_D_edge, 1, 1, 0) = -du;
      else
      {
        const auto d_map = imap_v<Scalar>(_D_edge.options()).intmd_expand(M);
        const auto diag_d = intmd_diagonalize(d_map);
        _J.d(_D_edge, 2, 1, 1) = (-du).intmd_unsqueeze(1) * diag_d;
      }
    }
  }
}
} // namespace neml2
