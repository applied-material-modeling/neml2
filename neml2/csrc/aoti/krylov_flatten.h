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

#pragma once

// Internal header -- NOT shipped. Glue between the per-group tensor lists the
// `NonlinearSystem` layer speaks (each DENSE group `(*B, group_total)`) and the
// single flat `(Bflat, N)` batch-of-column-vectors the Krylov solvers (krylov.h)
// operate on. Header-only, shared by the AOTI + eager Krylov systems and the
// unit test.
//
// v1 scope: DENSE groups only. A BLOCK (sub-batch) unknown/residual group has no
// canonical square linearization here, so `assert_all_dense` rejects it up front
// with a clear FatalError -- iterative solvers are deferred for sub-batch models
// (crystal-plasticity Schur), the same class of limitation as cpp-eager.

#include <cstdint>
#include <numeric>
#include <vector>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/assertions.h"
#include "neml2/csrc/aoti/nonlinear_system.h"

namespace neml2::aoti
{
/// Layout captured at flatten time so the inverse (unflatten) reproduces the
/// exact per-group shapes. `batch_shape` is the shared dynamic-batch prefix;
/// `widths[k]` is group k's trailing (DENSE) width; `N = sum(widths)`.
struct FlatSpec
{
  std::vector<int64_t> batch_shape;
  std::vector<int64_t> widths;
  int64_t N = 0;
};

/// Throw a FatalError if any group is not DENSE (iterative solvers require dense
/// unknown/residual groups in v1).
inline void
assert_all_dense(const std::vector<GroupLayout> & layout, const char * side)
{
  for (std::size_t k = 0; k < layout.size(); ++k)
    _assert(layout[k].structure != "block",
            "iterative (Krylov) linear solver requires dense ",
            side,
            " groups, but group ",
            k,
            " is BLOCK (sub-batch). Sub-batch models (e.g. crystal plasticity) "
            "are not yet supported with an iterative solver; use a direct linear "
            "solver (DenseLU / SchurComplement) for this model.");
}

/// Fold a list of DENSE per-group tensors `(*B, w_k)` into one flat `(Bflat, N)`
/// batch of column vectors, recording the shapes in `spec_out`.
inline at::Tensor
flatten_dense(const std::vector<at::Tensor> & groups,
              const std::vector<GroupLayout> & layout,
              FlatSpec & spec_out)
{
  _assert(groups.size() == layout.size(),
          "flatten_dense: tensor count ",
          groups.size(),
          " != group count ",
          layout.size());
  assert_all_dense(layout, "unknown/residual");
  _assert(!groups.empty(), "flatten_dense: no groups");

  // Shared dynamic-batch prefix = all but the trailing (DENSE) axis of group 0.
  const auto sizes0 = groups[0].sizes();
  spec_out.batch_shape.assign(sizes0.begin(), sizes0.end() - 1);
  int64_t bflat = 1;
  for (auto s : spec_out.batch_shape)
    bflat *= s;

  spec_out.widths.clear();
  std::vector<at::Tensor> flats;
  flats.reserve(groups.size());
  for (const auto & g : groups)
  {
    _assert(g.dim() >= 1, "flatten_dense: group tensor is a scalar");
    const int64_t w = g.size(-1);
    spec_out.widths.push_back(w);
    flats.push_back(g.reshape({bflat, w}));
  }
  spec_out.N = std::accumulate(spec_out.widths.begin(), spec_out.widths.end(), int64_t{0});
  return at::cat(flats, /*dim=*/1); // (Bflat, N)
}

/// Inverse of `flatten_dense`: split a flat `(Bflat, N)` back into per-group
/// tensors `(*B, w_k)` using the recorded `spec`.
inline std::vector<at::Tensor>
unflatten_dense(const at::Tensor & flat, const FlatSpec & spec)
{
  std::vector<at::Tensor> out;
  out.reserve(spec.widths.size());
  int64_t o = 0;
  for (const auto w : spec.widths)
  {
    auto piece = flat.slice(1, o, o + w); // (Bflat, w)
    std::vector<int64_t> shape = spec.batch_shape;
    shape.push_back(w);
    out.push_back(piece.reshape(shape).contiguous());
    o += w;
  }
  return out;
}
} // namespace neml2::aoti
