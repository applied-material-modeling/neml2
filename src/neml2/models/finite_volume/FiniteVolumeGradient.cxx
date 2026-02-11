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

#include "neml2/models/finite_volume/FiniteVolumeGradient.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{
register_NEML2_object(FiniteVolumeGradient);

OptionSet
FiniteVolumeGradient::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Compute prefactor-weighted gradients at cell edges using first-order reconstruction.";

  options.set_input("u") = VariableName(STATE, "u");
  options.set("u").doc() = "Cell-averaged field values.";

  options.set_parameter<TensorName<Scalar>>("prefactor") = "1";
  options.set("prefactor").doc() = "Cell-edge prefactor values (defaults to 1).";

  options.set_parameter<TensorName<Scalar>>("dx");
  options.set("dx").doc() = "Cell center spacing between adjacent cells.";

  options.set_output("grad_u") = VariableName(STATE, "grad_u");
  options.set("grad_u").doc() = "Cell-edge prefactor-weighted gradients.";

  return options;
}

FiniteVolumeGradient::FiniteVolumeGradient(const OptionSet & options)
  : Model(options),
    _u(declare_input_variable<Scalar>("u")),
    _prefactor(declare_parameter<Scalar>("prefactor", "prefactor", true)),
    _dx(declare_parameter<Scalar>("dx", "dx", true)),
    _grad_u(declare_output_variable<Scalar>("grad_u"))
{
}

void
FiniteVolumeGradient::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto N = _u.intmd_size(-1);
  const auto M = N - 1;
  const auto u_left = _u().intmd_slice(-1, indexing::Slice(0, N - 1));
  const auto u_right = _u().intmd_slice(-1, indexing::Slice(1, N));
  const auto prefactor_vec =
      (_prefactor.intmd_dim() == 0 ? _prefactor.intmd_expand(M) : _prefactor);
  const auto dx_vec = (_dx.intmd_dim() == 0 ? _dx.intmd_expand(M) : _dx);
  const auto inv_dx = 1.0 / dx_vec;

  const auto du = u_right - u_left;

  if (out)
    _grad_u = -prefactor_vec * du * inv_dx;

  if (dout_din)
  {
    // Build selector matrices for adjacent pairs. S_left picks i, S_right picks i+1.
    const auto u_map = imap_v<Scalar>(_u.options()).intmd_expand(N);
    const auto diag_u = intmd_diagonalize(u_map);
    const auto S_left = diag_u.intmd_slice(0, indexing::Slice(0, N - 1));
    const auto S_right = diag_u.intmd_slice(0, indexing::Slice(1, N));

    // dgrad_u/du = -prefactor / dx * (S_right - S_left)
    if (_u.is_dependent())
      _grad_u.d(_u, 2, 1, 1) = (-prefactor_vec * inv_dx).intmd_unsqueeze(1) * (S_right - S_left);

    // dgrad_u/dprefactor = -du / dx * I
    if (const auto * const prefactor = nl_param("prefactor"))
    {
      const auto dgrad_dprefactor = -du * inv_dx;
      if (prefactor->intmd_dim() == 0)
        _grad_u.d(*prefactor, 1, 1, 0) = dgrad_dprefactor;
      else
      {
        const auto d_map = imap_v<Scalar>(prefactor->options()).intmd_expand(M);
        const auto diag_d = intmd_diagonalize(d_map);
        _grad_u.d(*prefactor, 2, 1, 1) = dgrad_dprefactor.intmd_unsqueeze(1) * diag_d;
      }
    }

    // dgrad_u/ddx = (-prefactor * du) * d(1/dx)/dx = prefactor * du / dx^2
    if (const auto * const dx = nl_param("dx"))
    {
      const auto inv_dx2 = inv_dx * inv_dx;
      const auto dgrad_over_dx2 = prefactor_vec * du * inv_dx2;
      if (dx->intmd_dim() == 0)
        _grad_u.d(*dx, 1, 1, 0) = dgrad_over_dx2;
      else
      {
        const auto dx_map = imap_v<Scalar>(dx->options()).intmd_expand(M);
        const auto diag_dx = intmd_diagonalize(dx_map);
        _grad_u.d(*dx, 2, 1, 1) = dgrad_over_dx2.intmd_unsqueeze(1) * diag_dx;
      }
    }
  }
}
} // namespace neml2
