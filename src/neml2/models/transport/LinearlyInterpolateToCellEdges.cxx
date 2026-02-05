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

#include "neml2/models/transport/LinearlyInterpolateToCellEdges.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"
#include "neml2/tensors/indexing.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(LinearlyInterpolateToCellEdges);

OptionSet
LinearlyInterpolateToCellEdges::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Linearly interpolate cell-centered values to cell edges using cell center and edge "
      "positions.";

  options.set_parameter<TensorName<Scalar>>("cell_values");
  options.set("cell_values").doc() = "Cell-centered values to interpolate.";

  options.set_parameter<TensorName<Scalar>>("cell_centers");
  options.set("cell_centers").doc() = "Cell center positions.";

  options.set_parameter<TensorName<Scalar>>("cell_edges");
  options.set("cell_edges").doc() = "Cell edge positions.";

  options.set_output("edge_values") = VariableName(STATE, "q_edge");
  options.set("edge_values").doc() = "Linearly interpolated cell-edge values.";

  return options;
}

LinearlyInterpolateToCellEdges::LinearlyInterpolateToCellEdges(const OptionSet & options)
  : Model(options),
    _cell_values(declare_parameter<Scalar>("cell_values", "cell_values", true)),
    _cell_centers(declare_parameter<Scalar>("cell_centers", "cell_centers", true)),
    _cell_edges(declare_parameter<Scalar>("cell_edges", "cell_edges", true)),
    _edge_values(declare_output_variable<Scalar>("edge_values"))
{
}

void
LinearlyInterpolateToCellEdges::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  using namespace indexing;

  const auto N = _cell_centers.intmd_dim() == 0
                     ? (_cell_edges.intmd_dim() == 0 ? _cell_values.intmd_size(-1)
                                                     : _cell_edges.intmd_size(-1) - 1)
                     : _cell_centers.intmd_size(-1);

  const auto q_vec = (_cell_values.intmd_dim() == 0 ? _cell_values.intmd_expand(N) : _cell_values);
  const auto x_vec =
      (_cell_centers.intmd_dim() == 0 ? _cell_centers.intmd_expand(N) : _cell_centers);
  const auto xe_full =
      (_cell_edges.intmd_dim() == 0 ? _cell_edges.intmd_expand(N + 1) : _cell_edges);

  neml_assert_dbg(x_vec.intmd_size(-1) == N,
                  "Cell center positions must have size ",
                  N,
                  " in the last intermediate dimension.");
  neml_assert_dbg(xe_full.intmd_size(-1) == N + 1,
                  "Cell edge positions must have size ",
                  N + 1,
                  " in the last intermediate dimension.");
  neml_assert_dbg(q_vec.intmd_size(-1) == N,
                  "Cell values must have size ",
                  N,
                  " in the last intermediate dimension.");

  const auto q_left = q_vec.intmd_slice(-1, Slice(0, N - 1));
  const auto q_right = q_vec.intmd_slice(-1, Slice(1, N));
  const auto x_left = x_vec.intmd_slice(-1, Slice(0, N - 1));
  const auto x_right = x_vec.intmd_slice(-1, Slice(1, N));
  const auto xe_vec = xe_full.intmd_slice(-1, Slice(1, N));

  const auto denom = x_right - x_left;
  const auto inv = 1.0 / denom;
  const auto w_left = (x_right - xe_vec) * inv;
  const auto w_right = (xe_vec - x_left) * inv;

  if (out)
    _edge_values = w_left * q_left + w_right * q_right;

  if (dout_din)
  {
    const auto du = q_right - q_left;
    const auto inv2 = inv * inv;

    if (const auto * const q = nl_param("cell_values"))
    {
      if (q->intmd_dim() == 0)
        _edge_values.d(*q, 1, 1, 0) = w_left + w_right;
      else
      {
        const auto q_map = imap_v<Scalar>(_cell_values.options()).intmd_expand(N);
        const auto diag_q = intmd_diagonalize(q_map);
        const auto S_left = diag_q.intmd_slice(0, Slice(0, N - 1));
        const auto S_right = diag_q.intmd_slice(0, Slice(1, N));
        _edge_values.d(*q, 2, 1, 1) =
            w_left.intmd_unsqueeze(1) * S_left + w_right.intmd_unsqueeze(1) * S_right;
      }
    }

    if (const auto * const x = nl_param("cell_centers"))
    {
      const auto dqe_dxl = -(x_right - xe_vec) * inv2 * du;
      const auto dqe_dxr = -(xe_vec - x_left) * inv2 * du;

      if (x->intmd_dim() == 0)
        _edge_values.d(*x, 1, 1, 0) = dqe_dxl + dqe_dxr;
      else
      {
        const auto x_map = imap_v<Scalar>(_cell_centers.options()).intmd_expand(N);
        const auto diag_x = intmd_diagonalize(x_map);
        const auto S_left = diag_x.intmd_slice(0, Slice(0, N - 1));
        const auto S_right = diag_x.intmd_slice(0, Slice(1, N));
        _edge_values.d(*x, 2, 1, 1) =
            dqe_dxl.intmd_unsqueeze(1) * S_left + dqe_dxr.intmd_unsqueeze(1) * S_right;
      }
    }

    if (const auto * const xe = nl_param("cell_edges"))
    {
      const auto dqe_dxe = inv * du;
      if (xe->intmd_dim() == 0)
        _edge_values.d(*xe, 1, 1, 0) = dqe_dxe;
      else
      {
        const auto xe_map = imap_v<Scalar>(_cell_edges.options()).intmd_expand(N + 1);
        const auto diag_xe = intmd_diagonalize(xe_map);
        const auto S_mid = diag_xe.intmd_slice(0, Slice(1, N));
        _edge_values.d(*xe, 2, 1, 1) = dqe_dxe.intmd_unsqueeze(1) * S_mid;
      }
    }
  }
}
} // namespace neml2
