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

// Internal header -- NOT shipped. Inline batch-axis slice/concat helpers shared
// by DispatchedModel and the C++ tests. Header-only so neither needs to link an
// exported symbol (keeps the libneml2.so ABI surface limited to the public
// Model / scheduler / DispatchedModel classes).
//
// Contract: a value map is keyed by variable name with each tensor shaped
// ``(*B, *base)``; the leading dim 0 is the dynamic batch axis. By the same
// convention the AOTI Model uses at its call boundary, an input is either
// fully batched (``size(0) == B``) or broadcast/unbatched (``size(0) == 1`` or
// rank 0). Slicing acts on dim 0; broadcast/unbatched entries pass through.

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/assertions.h"

namespace neml2::aoti
{
/// Batch size B = the dim-0 extent of any fully-batched entry (1 if there are
/// none). Broadcast/unbatched entries (rank 0 or dim-0 size 1) do not raise it.
inline int64_t
infer_batch_size(const std::map<std::string, at::Tensor> & m)
{
  int64_t b = 1;
  for (const auto & [name, t] : m)
    if (t.dim() >= 1)
      b = std::max(b, t.size(0));
  return b;
}

/// Narrow every fully-batched entry to ``[start, start + count)`` along dim 0;
/// broadcast/unbatched entries (dim-0 size 1 or rank 0) pass through whole.
inline std::map<std::string, at::Tensor>
slice_batch(const std::map<std::string, at::Tensor> & m, int64_t start, int64_t count)
{
  std::map<std::string, at::Tensor> out;
  for (const auto & [name, t] : m)
  {
    if (t.dim() >= 1 && t.size(0) != 1)
      out.emplace(name, t.narrow(0, start, count));
    else
      out.emplace(name, t);
  }
  return out;
}

/// Select rows ``idx`` (a 1-D int64 index tensor) along dim 0 from every entry
/// with a batch axis (rank >= 1). The arbitrary-index counterpart of
/// ``slice_batch`` (which narrows a contiguous range) -- used by the substep
/// masking driver to solve only the still-unconverged subset of the dynamic
/// batch. Unlike ``slice_batch`` there is NO size-1 broadcast passthrough: the
/// masking driver's state entries are all materialized at the full batch size
/// ``B`` (``_prepare_inputs`` broadcasts them), so a size-1 dim-0 is a genuine
/// ``B == 1`` batch that must be indexed, not a broadcast. Inverse:
/// ``scatter_batch_``.
inline std::map<std::string, at::Tensor>
index_select_batch(const std::map<std::string, at::Tensor> & m, const at::Tensor & idx)
{
  std::map<std::string, at::Tensor> out;
  for (const auto & [name, t] : m)
  {
    if (t.dim() >= 1)
      out.emplace(name, t.index_select(0, idx).contiguous());
    else
      out.emplace(name, t);
  }
  return out;
}

/// Write the rows of ``sub`` (aligned with ``idx``) into ``dst`` in place along
/// dim 0 at positions ``idx``. Only keys present in both (with rank >= 1) are
/// scattered. ``dst`` is mutated (the substep driver owns full-batch
/// accumulators it scatters converged rows into). Companion of
/// ``index_select_batch`` -- same no-broadcast-passthrough contract.
inline void
scatter_batch_(std::map<std::string, at::Tensor> & dst,
               const at::Tensor & idx,
               const std::map<std::string, at::Tensor> & sub)
{
  for (auto & [name, t] : dst)
  {
    auto sit = sub.find(name);
    if (sit != sub.end() && t.dim() >= 1)
      t.index_copy_(0, idx, sit->second);
  }
}

/// Concatenate per-chunk value maps along dim 0, keyed by the first chunk's
/// names. Every chunk must carry every key.
inline std::map<std::string, at::Tensor>
cat_batch(const std::vector<std::map<std::string, at::Tensor>> & chunks)
{
  _assert(!chunks.empty(), "cat_batch: no chunks to concatenate.");
  if (chunks.size() == 1)
    return chunks.front();

  std::map<std::string, at::Tensor> out;
  for (const auto & [name, first] : chunks.front())
  {
    (void)first;
    std::vector<at::Tensor> parts;
    parts.reserve(chunks.size());
    for (const auto & c : chunks)
    {
      auto it = c.find(name);
      _assert(it != c.end(), "cat_batch: key '", name, "' is missing from one of the chunks.");
      parts.push_back(it->second);
    }
    out.emplace(name, at::cat(parts, /*dim=*/0));
  }
  return out;
}

/// Concatenate plain per-chunk tensors (e.g. the assembled Jacobian) along
/// dim 0.
inline at::Tensor
cat_batch_tensor(const std::vector<at::Tensor> & parts)
{
  _assert(!parts.empty(), "cat_batch_tensor: no parts to concatenate.");
  if (parts.size() == 1)
    return parts.front();
  return at::cat(parts, /*dim=*/0);
}

/// Move every entry of a value map onto ``device`` (no-op for entries already
/// there).
inline std::map<std::string, at::Tensor>
to_device(const std::map<std::string, at::Tensor> & m, at::Device device)
{
  std::map<std::string, at::Tensor> out;
  for (const auto & [name, t] : m)
    out.emplace(name, t.device() == device ? t : t.to(device));
  return out;
}

/// Move every leaf block of a nested variable-pair Jacobian
/// (``{out_name: {in_name: block}}``) onto ``device``.
inline std::map<std::string, std::map<std::string, at::Tensor>>
to_device_nested(const std::map<std::string, std::map<std::string, at::Tensor>> & j,
                 at::Device device)
{
  std::map<std::string, std::map<std::string, at::Tensor>> out;
  for (const auto & [o, row] : j)
    for (const auto & [i, t] : row)
      out[o].emplace(i, t.device() == device ? t : t.to(device));
  return out;
}

/// Concatenate per-chunk nested Jacobians, passing batch-independent blocks
/// through instead of concatenating them.
///
/// A batch-independent block (e.g. a constant stiffness tensor) is returned by
/// the per-device model unbatched at its natural ``(*out_base, *in_base)`` shape
/// -- identical across every chunk. Concatenating it along dim 0 would corrupt
/// it (stacking N identical copies and mislabelling an ``out_base`` axis as
/// batch). We detect it structurally: a block with exactly
/// ``out_base_ndim[o] + in_base_ndim[i]`` dims carries no leading batch axis, so
/// take the first chunk's copy; any block with more dims is batched and is
/// concatenated as usual.
inline std::map<std::string, std::map<std::string, at::Tensor>>
cat_batch_nested(
    const std::vector<std::map<std::string, std::map<std::string, at::Tensor>>> & chunks,
    const std::map<std::string, int64_t> & out_base_ndim,
    const std::map<std::string, int64_t> & in_base_ndim)
{
  _assert(!chunks.empty(), "cat_batch_nested: no chunks to concatenate.");
  if (chunks.size() == 1)
    return chunks.front();

  std::map<std::string, std::map<std::string, at::Tensor>> out;
  for (const auto & [o, row] : chunks.front())
    for (const auto & [i, first] : row)
    {
      const int64_t trail = out_base_ndim.at(o) + in_base_ndim.at(i);
      if (first.dim() == trail)
      {
        // Batch-independent: identical across chunks, take the first.
        out[o].emplace(i, first);
        continue;
      }
      std::vector<at::Tensor> parts;
      parts.reserve(chunks.size());
      for (const auto & c : chunks)
      {
        auto oit = c.find(o);
        _assert(oit != c.end(), "cat_batch_nested: output '", o, "' missing from a chunk.");
        auto iit = oit->second.find(i);
        _assert(iit != oit->second.end(),
                "cat_batch_nested: block ('",
                o,
                "', '",
                i,
                "') missing from a chunk.");
        parts.push_back(iit->second);
      }
      out[o].emplace(i, at::cat(parts, /*dim=*/0));
    }
  return out;
}
} // namespace neml2::aoti
