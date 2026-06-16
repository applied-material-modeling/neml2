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

#include "neml2/csrc/aoti/Model.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ATen.h is the umbrella header that pulls in the full at::Tensor template
// definitions (Tensor::item<T>(), etc.). The narrower <ATen/Functions.h>
// alone leaves Tensor declared but its member-templates unparseable -- a
// regression we hit when aoti stopped inheriting the legacy misc/tensor
// PCH that used to drag the full umbrella in.
#include <ATen/ATen.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace neml2::aoti
{
namespace
{
// Local check-and-throw helper. Kept private to this TU so the AOTI
// submodule does not depend on the legacy `neml2/misc/assertions.h` --
// `aoti::Model` is intentionally free-standing (no link to libneml2_misc /
// libneml2_tensor), so it can outlive the rest of the C++ object tower
// once  completes. The streaming-args ergonomics match the
// neml_assert style at every call site below.
template <typename... Args>
[[noreturn]] void
_throw(const Args &... args)
{
  std::ostringstream oss;
  ((oss << args), ...);
  throw std::runtime_error(oss.str());
}

template <typename... Args>
inline void
_assert(bool cond, const Args &... args)
{
  if (!cond)
    _throw(args...);
}

// Build a loader with the consistent constructor args used across all AOTI
// artifacts produced by neml2-compile.
std::unique_ptr<torch::inductor::AOTIModelPackageLoader>
make_loader(const std::filesystem::path & path)
{
  return std::make_unique<torch::inductor::AOTIModelPackageLoader>(path.string(),
                                                                   /*model_name=*/"model",
                                                                   /*run_single_threaded=*/false,
                                                                   /*num_runners=*/1,
                                                                   /*device_index=*/-1);
}

at::Device
parse_device(const std::string & s)
{
  if (s == "cpu")
    return at::kCPU;
  if (s == "cuda")
    return at::kCUDA;
  _assert(false, "aoti::Model: unsupported device '", s, "' in metadata.");
  return at::kCPU;
}

at::ScalarType
parse_dtype(const std::string & s)
{
  if (s == "float64")
    return at::kDouble;
  if (s == "float32")
    return at::kFloat;
  _assert(false, "aoti::Model: unsupported dtype '", s, "' in metadata.");
  return at::kDouble;
}

// Bool convergence check: max-norm over the batch must clear (||b|| < atol)
// or (||b||/||b0|| < rtol). Single GPU->CPU sync via .item<bool>().
bool
check_converged(const at::Tensor & b_norm, const at::Tensor & b0_norm, double atol, double rtol)
{
  return at::all(at::logical_or(b_norm < atol, b_norm / b0_norm < rtol)).item<bool>();
}

// Tri-state stop check for the Newton loop. Folds the divergence detection
// (any NaN / Inf in the residual norm) into the same device → host sync as
// the existing convergence check, so on CUDA we still pay only one d2h per
// iteration. The extra cost is a single `isfinite` reduction kernel, which
// is negligible next to the step graph it runs alongside.
enum class StopStatus
{
  Continue = 0,
  Converged = 1,
  Diverged = 2,
};

inline StopStatus
check_stop(const at::Tensor & b_norm, const at::Tensor & b0_norm, double atol, double rtol)
{
  auto diverged = at::any(at::logical_not(at::isfinite(b_norm)));
  auto converged = at::all(at::logical_or(b_norm < atol, b_norm / b0_norm < rtol));
  // Stack on-device into a (2,) bool tensor and pull both bits over in one
  // d2h memcpy. The two reductions feed the same downstream sync, so this is
  // not strictly more sync than the original single-condition check.
  auto packed = at::stack({diverged, converged}).cpu();
  if (packed[0].item<bool>())
    return StopStatus::Diverged;
  if (packed[1].item<bool>())
    return StopStatus::Converged;
  return StopStatus::Continue;
}
} // namespace

Model::Model(const std::filesystem::path & meta_path)
{
  std::ifstream meta_f(meta_path);
  _assert(static_cast<bool>(meta_f),
          "aoti::Model: failed to open metadata file '",
          meta_path.string(),
          "'. Path must point at the _meta.json written by `neml2-compile`.");
  const auto meta = nlohmann::json::parse(meta_f);

  // Schema-version handshake. v5 moves implicit-segment I/O from a
  // single packed (*dyn, u_size) flat slab to per-variable tensors in
  // their natural (*dyn, *sub_batch, *base) shape; the per-segment
  // metadata records per-variable sub_batch_shape / base_shape so the
  // C++ side can pack the segment loader's positional inputs and so
  // convergence-norm bookkeeping can sum per-variable contributions.
  // Older caches must be regenerated; we fail loudly instead of
  // silently misinterpreting them.
  // dependencies: aoti.schema_version
  static constexpr int kSupportedSchemaVersion = 3;
  const auto schema_version = meta.value("schema_version", 0);
  _assert(schema_version == kSupportedSchemaVersion,
          "aoti::Model: metadata schema_version=",
          schema_version,
          " in '",
          meta_path.string(),
          "' is not compatible with the build's supported version=",
          kSupportedSchemaVersion,
          ". Regenerate the artifact via `neml2-compile`.");

  const auto kind = meta.value("type", std::string{});
  _assert(kind == "composed",
          "aoti::Model: unsupported metadata 'type' '",
          kind,
          "' in '",
          meta_path.string(),
          "'. Expected 'composed'.");

  // Device + dtype baked into the artifact at export time.
  _device = parse_device(meta.value("device", std::string("cpu")));
  _dtype = parse_dtype(meta.value("dtype", std::string("float64")));

  // .pt2 basenames are resolved against the directory holding the metadata
  // file itself.
  const auto cache_dir = meta_path.parent_path();

  // Master inputs / outputs.
  _input_names.reserve(meta["inputs"].size());
  _input_sizes.reserve(meta["inputs"].size());
  _input_offsets.reserve(meta["inputs"].size());
  for (const auto & v : meta["inputs"])
  {
    const auto name = v["name"].get<std::string>();
    const auto sz = v["var_size"].get<int>();
    _input_names.push_back(name);
    _input_offsets.push_back(_input_total_size);
    _input_sizes.push_back(sz);
    _input_total_size += sz;
  }
  _output_names.reserve(meta["outputs"].size());
  _output_sizes.reserve(meta["outputs"].size());
  for (const auto & v : meta["outputs"])
  {
    _output_names.push_back(v["name"].get<std::string>());
    _output_sizes.push_back(v["var_size"].get<int>());
  }

  // Promoted parameters (v2; empty in the common fully-baked case).
  if (meta.contains("parameters"))
  {
    for (const auto & p : meta["parameters"])
    {
      const auto name = p["name"].get<std::string>();
      const auto dtype = parse_dtype(p.value("dtype", std::string("float64")));
      const auto shape = p["shape"].get<std::vector<int64_t>>();
      const auto device =
          p.contains("device") ? parse_device(p["device"].get<std::string>()) : _device;
      const auto opts = at::TensorOptions().dtype(dtype).device(device);

      // Inline values vs. spilled state-dict reference.
      at::Tensor t;
      if (p.contains("values"))
      {
        const auto values = p["values"].get<std::vector<double>>();
        // Build on CPU first, then move to the recorded device. CUDA-side
        // tensor construction from a std::vector is awkward; CPU + .to() is
        // the standard pattern and keeps the loader-side simple.
        const auto flat = at::from_blob(const_cast<double *>(values.data()),
                                        {static_cast<int64_t>(values.size())},
                                        at::TensorOptions().dtype(at::kDouble))
                              .clone();
        t = flat.to(opts).reshape(shape);
      }
      else
      {
        // values_in / values_key spill: deferred until we actually exercise
        // tensors > 1 MiB. The Python exporter inlines everything for now.
        _assert(false,
                "aoti::Model: parameter '",
                name,
                "' uses the values_in spill format which is not yet "
                "supported on the C++ side. Re-export with smaller tensors.");
      }
      _named_parameters.emplace(name, t.contiguous());
    }
  }

  // Segments.
  _segments.reserve(meta["segments"].size());
  for (std::size_t i = 0; i < meta["segments"].size(); ++i)
  {
    const auto & seg_meta = meta["segments"][i];
    const auto seg_kind = seg_meta.value("kind", std::string{});
    Segment seg;

    // v2: promoted-parameter tail consumed by this segment's graphs. Empty in
    // the fully-baked case; otherwise lists names in graph-call order.
    if (seg_meta.contains("param_inputs"))
      seg.param_inputs = seg_meta["param_inputs"].get<std::vector<std::string>>();

    auto parse_var_info = [](const nlohmann::json & v) -> Segment::VarInfo
    {
      Segment::VarInfo info;
      info.name = v["name"].get<std::string>();
      info.var_size = v["var_size"].get<int>();
      if (v.contains("sub_batch_shape"))
        info.sub_batch_shape = v["sub_batch_shape"].get<std::vector<int64_t>>();
      if (v.contains("base_shape"))
        info.base_shape = v["base_shape"].get<std::vector<int64_t>>();
      return info;
    };

    if (seg_kind == "forward")
    {
      seg.kind = SegmentKind::Forward;
      seg.fwd_loader = make_loader(cache_dir / seg_meta["package"].get<std::string>());
      if (seg_meta.contains("jvp_package"))
        seg.jvp_loader = make_loader(cache_dir / seg_meta["jvp_package"].get<std::string>());
      for (const auto & v : seg_meta["inputs"])
        seg.fwd_inputs.push_back(v["name"].get<std::string>());
      for (const auto & v : seg_meta["outputs"])
        seg.fwd_outputs.push_back(v["name"].get<std::string>());
      // v7 per-(out_var, in_var) Jacobian-pair metadata. Present iff
      // the segment also packaged a JVP loader; orders one entry per
      // trailing pair tensor in the JVP loader's output tuple
      // (row-major: outputs outer, structural inputs inner).
      if (seg_meta.contains("jacobian_pairs"))
      {
        for (const auto & p : seg_meta["jacobian_pairs"])
        {
          Segment::PairInfo pi;
          pi.out_var = p["out_var"].get<std::string>();
          pi.in_var = p["in_var"].get<std::string>();
          if (p.contains("out_base_shape"))
            pi.out_base_shape = p["out_base_shape"].get<std::vector<int64_t>>();
          if (p.contains("in_base_shape"))
            pi.in_base_shape = p["in_base_shape"].get<std::vector<int64_t>>();
          if (p.contains("in_sub_batch_shape"))
            pi.in_sub_batch_shape = p["in_sub_batch_shape"].get<std::vector<int64_t>>();
          seg.jacobian_pairs.push_back(std::move(pi));
        }
      }
    }
    else if (seg_kind == "implicit")
    {
      seg.kind = SegmentKind::Implicit;
      seg.rhs_loader = make_loader(cache_dir / seg_meta["rhs_package"].get<std::string>());
      seg.step_loader = make_loader(cache_dir / seg_meta["step_package"].get<std::string>());
      if (seg_meta.contains("ift_package"))
        seg.ift_loader = make_loader(cache_dir / seg_meta["ift_package"].get<std::string>());

      for (const auto & v : seg_meta["unknowns"])
        seg.unknowns.push_back(parse_var_info(v));
      for (const auto & v : seg_meta["givens"])
        seg.givens.push_back(parse_var_info(v));
      for (const auto & v : seg_meta["residuals"])
        seg.residuals.push_back(parse_var_info(v));

      // v7 per-group metadata for the per-group I/O contract. Newton
      // inner loop runs entirely per-group; per-var ↔ per-group conv
      // happens twice per solve via _pack_groups / _unpack_groups.
      auto parse_group_info = [&parse_var_info](const nlohmann::json & g) -> Segment::GroupInfo
      {
        Segment::GroupInfo gi;
        gi.structure = g.value("structure", std::string("dense"));
        if (g.contains("sub_batch_shape"))
          gi.sub_batch_shape = g["sub_batch_shape"].get<std::vector<int64_t>>();
        if (g.contains("per_var_info"))
        {
          for (const auto & v : g["per_var_info"])
            gi.per_var_info.push_back(parse_var_info(v));
        }
        return gi;
      };
      for (const auto & g : seg_meta["unknown_group_infos"])
        seg.unknown_groups.push_back(parse_group_info(g));
      for (const auto & g : seg_meta["given_group_infos"])
        seg.given_groups.push_back(parse_group_info(g));
      for (const auto & g : seg_meta["residual_group_infos"])
        seg.residual_groups.push_back(parse_group_info(g));

      // v7 per-(row_group, col_group) IFT cell metadata.
      // ift_cells[k] corresponds to ift_outs[k] in row-major order.
      if (seg_meta.contains("ift_cells"))
      {
        for (const auto & c : seg_meta["ift_cells"])
        {
          Segment::CellInfo ci;
          ci.row_group_idx = c["row_group_idx"].get<int64_t>();
          ci.col_group_idx = c["col_group_idx"].get<int64_t>();
          ci.row_structure = c["row_structure"].get<std::string>();
          ci.col_structure = c["col_structure"].get<std::string>();
          if (c.contains("sub_batch_shape"))
            ci.sub_batch_shape = c["sub_batch_shape"].get<std::vector<int64_t>>();
          if (c.contains("row_vars"))
          {
            for (const auto & v : c["row_vars"])
              ci.row_vars.push_back(parse_var_info(v));
          }
          if (c.contains("col_vars"))
          {
            for (const auto & v : c["col_vars"])
              ci.col_vars.push_back(parse_var_info(v));
          }
          seg.ift_cells.push_back(std::move(ci));
        }
      }
      seg.atol = seg_meta["atol"].get<double>();
      seg.rtol = seg_meta["rtol"].get<double>();
      seg.miters = static_cast<std::size_t>(seg_meta["miters"].get<int>());

      // Optional linesearch block. Absent for plain Newton; present and
      // populated for NewtonWithLineSearch.
      if (seg_meta.contains("linesearch"))
      {
        const auto & ls = seg_meta["linesearch"];
        seg.ls_type = ls.value("type", std::string("BACKTRACKING"));
        seg.ls_max_iters = static_cast<std::size_t>(ls.value("max_iters", 1));
        seg.ls_cutback = ls.value("cutback", 2.0);
        seg.ls_c = ls.value("c", 1.0e-3);
      }
      _assert(!seg.unknowns.empty(),
              "aoti::Model: implicit segment ",
              i,
              " has zero unknowns -- an implicit system needs at least one "
              "unknown to solve for.");

      if (seg_meta.contains("predictor_package"))
      {
        seg.predictor_loader =
            make_loader(cache_dir / seg_meta["predictor_package"].get<std::string>());
        for (const auto & v : seg_meta["predictor_inputs"])
          seg.predictor_inputs.push_back(v["name"].get<std::string>());
        for (const auto & v : seg_meta["predictor_outputs"])
          seg.predictor_outputs.push_back(v["name"].get<std::string>());
      }
    }
    else
    {
      _assert(false,
              "aoti::Model: unknown segment kind '",
              seg_kind,
              "' at segment index ",
              i,
              " in '",
              meta_path.string(),
              "'.");
    }
    _segments.push_back(std::move(seg));
  }
}

Model::~Model() = default;

const std::vector<std::string> &
Model::input_names() const noexcept
{
  return _input_names;
}

const std::vector<std::string> &
Model::output_names() const noexcept
{
  return _output_names;
}

const std::vector<int> &
Model::input_sizes() const noexcept
{
  return _input_sizes;
}

const std::vector<int> &
Model::output_sizes() const noexcept
{
  return _output_sizes;
}

std::map<std::string, at::Tensor> &
Model::named_parameters() noexcept
{
  return _named_parameters;
}

const std::map<std::string, at::Tensor> &
Model::named_parameters() const noexcept
{
  return _named_parameters;
}

std::vector<at::Tensor>
Model::_gather_params(const std::vector<std::string> & names) const
{
  std::vector<at::Tensor> out;
  out.reserve(names.size());
  for (const auto & n : names)
  {
    auto it = _named_parameters.find(n);
    _assert(it != _named_parameters.end(),
            "aoti::Model: segment references promoted parameter '",
            n,
            "' which is missing from named_parameters(). Either the user mutated the map "
            "or the metadata is inconsistent.");
    out.push_back(it->second.contiguous());
  }
  return out;
}

// ---------------------------------------------------------------------------
// Public ops: forward / jvp / jacobian
// ---------------------------------------------------------------------------

std::map<std::string, at::Tensor>
Model::forward(const std::map<std::string, at::Tensor> & inputs) const
{
  std::map<std::string, at::Tensor> state;
  for (const auto & n : _input_names)
  {
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::forward: missing required input '", n, "'.");
    state[n] = it->second.contiguous();
  }

  for (const auto & seg : _segments)
  {
    if (seg.kind == SegmentKind::Forward)
    {
      _run_forward_segment(seg, state);
    }
    else
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
      _run_implicit_segment(seg, state, u_solved_groups, g_groups);
    }
  }

  std::map<std::string, at::Tensor> outputs;
  for (const auto & n : _output_names)
  {
    auto it = state.find(n);
    _assert(it != state.end(),
            "aoti::Model::forward: output '",
            n,
            "' was not produced by any segment.");
    outputs[n] = it->second;
  }
  return outputs;
}

std::pair<std::map<std::string, at::Tensor>, at::Tensor>
Model::jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  // Pack state from caller inputs.
  std::map<std::string, at::Tensor> state;
  for (const auto & n : _input_names)
  {
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::jacobian: missing required input '", n, "'.");
    state[n] = it->second.contiguous();
  }

  // dstate[var] is (*B, var_size, M) where M = _input_total_size. Initialized
  // with identity columns for each master input, then composed segment-by-
  // segment in lockstep with `state`.
  const auto & ref = state.at(_input_names.front());
  // Infer batch shape from a master input. We assume the input arrives as
  // (*B, *base_shape) where base_shape's product == its `var_size`. The
  // simplest heuristic: the trailing axes match `var_size`, so split off the
  // leading axes as batch. Match the size exactly to disambiguate.
  const auto in_sz = _input_sizes.front();
  std::vector<int64_t> batch_shape_vec;
  {
    int64_t trail = 1;
    int64_t cut = ref.dim();
    while (cut > 0 && trail < in_sz)
    {
      --cut;
      trail *= ref.size(cut);
    }
    _assert(trail == in_sz,
            "aoti::Model::jacobian: could not isolate batch shape from input '",
            _input_names.front(),
            "' (var_size=",
            in_sz,
            ", tensor sizes=",
            ref.sizes(),
            ").");
    for (int64_t d = 0; d < cut; ++d)
      batch_shape_vec.push_back(ref.size(d));
  }
  at::IntArrayRef batch_shape(batch_shape_vec);

  std::map<std::string, at::Tensor> dstate;
  _init_dstate(ref.options(), batch_shape, dstate);

  for (const auto & seg : _segments)
  {
    if (seg.kind == SegmentKind::Forward)
    {
      _assert(static_cast<bool>(seg.jvp_loader),
              "aoti::Model::jacobian: forward segment is missing its jvp_package loader. "
              "Re-export via `neml2-compile`.");
      _run_forward_segment_jacobian(seg, state, dstate);
    }
    else
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
      _run_implicit_segment(seg, state, u_solved_groups, g_groups);
      _assert(static_cast<bool>(seg.ift_loader),
              "aoti::Model::jacobian: implicit segment is missing its ift_package loader. "
              "Re-export via `neml2-compile`.");
      _run_implicit_segment_jacobian(seg, u_solved_groups, g_groups, dstate);
    }
  }

  // Pack outputs and assembled J of shape (*B, sum(out), sum(in)).
  std::map<std::string, at::Tensor> outputs;
  for (const auto & n : _output_names)
    outputs[n] = state.at(n);

  std::vector<at::Tensor> out_blocks;
  out_blocks.reserve(_output_names.size());
  for (const auto & n : _output_names)
  {
    auto it = dstate.find(n);
    _assert(it != dstate.end(),
            "aoti::Model::jacobian: dstate missing entry for master output '",
            n,
            "'.");
    out_blocks.push_back(it->second);
  }
  auto J = at::cat(out_blocks, /*dim=*/-2).contiguous();

  return {std::move(outputs), std::move(J)};
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::jvp(const std::map<std::string, at::Tensor> & inputs,
           const std::map<std::string, at::Tensor> & tangents) const
{
  // Compose against the full Jacobian. For the typical use case (single
  // batched tangent vector) this is the right answer: J @ v where v is the
  // input tangent stacked in declared order. Implementing this on top of
  // jacobian() trades some performance for a single, well-tested code path.
  auto [outputs, J] = jacobian(inputs);

  // Pack the tangent vector v: shape (*B, sum(in_sizes)). Missing tangents
  // default to zero.
  std::vector<at::Tensor> v_parts;
  v_parts.reserve(_input_names.size());
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = tangents.find(n);
    at::Tensor part;
    if (it != tangents.end())
    {
      // Flatten trailing base axes into a single var_size slot.
      const auto sz = _input_sizes[k];
      auto t = it->second;
      int64_t trail = 1;
      int64_t cut = t.dim();
      while (cut > 0 && trail < sz)
      {
        --cut;
        trail *= t.size(cut);
      }
      _assert(trail == sz,
              "aoti::Model::jvp: tangent for input '",
              n,
              "' has shape ",
              t.sizes(),
              " that does not collapse to var_size=",
              sz);
      std::vector<int64_t> new_shape;
      for (int64_t d = 0; d < cut; ++d)
        new_shape.push_back(t.size(d));
      new_shape.push_back(sz);
      part = t.reshape(new_shape).to(_dtype);
    }
    else
    {
      // Zero tangent; piggyback on J's leading batch shape (and trailing in dim).
      const auto sz = _input_sizes[k];
      std::vector<int64_t> shape(J.sizes().begin(), J.sizes().end() - 2);
      shape.push_back(sz);
      part = at::zeros(shape, J.options());
    }
    v_parts.push_back(part.contiguous());
  }
  auto v = at::cat(v_parts, /*dim=*/-1).unsqueeze(-1); // (*B, sum(in), 1)
  auto Jv = at::matmul(J, v).squeeze(-1);              // (*B, sum(out))

  // Split Jv back per output (kept flat-trailing; caller can reshape via
  // output_sizes() if needed).
  std::map<std::string, at::Tensor> jvp_outputs;
  int64_t offset = 0;
  for (std::size_t i = 0; i < _output_names.size(); ++i)
  {
    const auto sz = _output_sizes[i];
    jvp_outputs[_output_names[i]] = Jv.narrow(-1, offset, sz).contiguous();
    offset += sz;
  }

  return {std::move(outputs), std::move(jvp_outputs)};
}

// ---------------------------------------------------------------------------
// Segment runners (value path)
// ---------------------------------------------------------------------------

void
Model::_run_forward_segment(const Segment & seg, std::map<std::string, at::Tensor> & state) const
{
  std::vector<at::Tensor> inputs;
  inputs.reserve(seg.fwd_inputs.size() + seg.param_inputs.size());
  for (const auto & name : seg.fwd_inputs)
  {
    auto it = state.find(name);
    _assert(it != state.end(),
            "aoti::Model: forward segment needs input '",
            name,
            "' which is not in the state map.");
    inputs.push_back(it->second.contiguous());
  }
  // Promoted-parameter tail.
  for (auto & p : _gather_params(seg.param_inputs))
    inputs.push_back(std::move(p));

  const auto outs = seg.fwd_loader->run(inputs);
  _assert(outs.size() == seg.fwd_outputs.size(),
          "aoti::Model: forward segment returned ",
          outs.size(),
          " tensors, expected ",
          seg.fwd_outputs.size());
  for (std::size_t i = 0; i < seg.fwd_outputs.size(); ++i)
    state[seg.fwd_outputs[i]] = outs[i];
}

std::vector<at::Tensor>
Model::_pack_groups(const std::map<std::string, at::Tensor> & state,
                    const std::vector<Segment::GroupInfo> & groups) const
{
  // Mirrors :meth:`AssembledVector.from_dict` -- for each group, cat
  // per-var contributions along the last axis. BLOCK groups keep the
  // group's sub_batch axes (each var's sub matches the group's by
  // construction); DENSE groups fold each var's sub into base then cat.
  std::vector<at::Tensor> packed;
  packed.reserve(groups.size());
  for (const auto & group : groups)
  {
    std::vector<at::Tensor> parts;
    parts.reserve(group.per_var_info.size());
    for (const auto & v : group.per_var_info)
    {
      auto it = state.find(v.name);
      _assert(
          it != state.end(), "aoti::Model::_pack_groups: state missing variable '", v.name, "'.");
      const auto & t = it->second;
      const int64_t var_trail =
          static_cast<int64_t>(v.sub_batch_shape.size() + v.base_shape.size());
      _assert(t.dim() >= var_trail,
              "aoti::Model::_pack_groups: variable '",
              v.name,
              "' ndim=",
              t.dim(),
              " < sub_batch+base ndim=",
              var_trail);
      int64_t base_total = 1;
      for (auto s : v.base_shape)
        base_total *= s;
      int64_t sub_total = 1;
      for (auto s : v.sub_batch_shape)
        sub_total *= s;
      const int64_t dyn_ndim = t.dim() - var_trail;
      std::vector<int64_t> target;
      target.reserve(static_cast<std::size_t>(dyn_ndim) + v.sub_batch_shape.size() + 1);
      for (int64_t d = 0; d < dyn_ndim; ++d)
        target.push_back(t.size(d));
      if (group.structure == "block")
      {
        for (auto s : v.sub_batch_shape)
          target.push_back(s);
        target.push_back(base_total);
      }
      else
      {
        target.push_back(sub_total * base_total);
      }
      parts.push_back(t.contiguous().reshape(target));
    }
    _assert(!parts.empty(),
            "aoti::Model::_pack_groups: encountered an empty group; v7 layouts "
            "always have at least one variable per declared group.");
    packed.push_back(at::cat(parts, /*dim=*/-1).contiguous());
  }
  return packed;
}

void
Model::_unpack_groups(const std::vector<at::Tensor> & group_tensors,
                      const std::vector<Segment::GroupInfo> & groups,
                      std::map<std::string, at::Tensor> & state) const
{
  _assert(group_tensors.size() == groups.size(),
          "aoti::Model::_unpack_groups: tensor count ",
          group_tensors.size(),
          " != group count ",
          groups.size());
  for (std::size_t gi = 0; gi < groups.size(); ++gi)
  {
    const auto & gt = group_tensors[gi];
    const auto & group = groups[gi];
    // BLOCK group tensor: (*B, *group_sub, group_base_total).
    // DENSE group tensor: (*B, group_total).
    const int64_t group_trail =
        (group.structure == "block") ? static_cast<int64_t>(group.sub_batch_shape.size()) + 1 : 1;
    _assert(gt.dim() >= group_trail,
            "aoti::Model::_unpack_groups: group tensor ndim=",
            gt.dim(),
            " < trail ndim=",
            group_trail);
    const int64_t dyn_ndim = gt.dim() - group_trail;
    std::vector<int64_t> batch_shape;
    batch_shape.reserve(static_cast<std::size_t>(dyn_ndim));
    for (int64_t d = 0; d < dyn_ndim; ++d)
      batch_shape.push_back(gt.size(d));

    int64_t offset = 0;
    for (const auto & v : group.per_var_info)
    {
      int64_t base_total = 1;
      for (auto s : v.base_shape)
        base_total *= s;
      int64_t sub_total = 1;
      for (auto s : v.sub_batch_shape)
        sub_total *= s;
      const int64_t segment_size =
          (group.structure == "block") ? base_total : (sub_total * base_total);
      auto part = gt.narrow(/*dim=*/-1, /*start=*/offset, /*length=*/segment_size);

      std::vector<int64_t> target = batch_shape;
      for (auto s : v.sub_batch_shape)
        target.push_back(s);
      for (auto s : v.base_shape)
        target.push_back(s);
      state[v.name] = part.reshape(target).contiguous();
      offset += segment_size;
    }
  }
}

void
Model::_run_implicit_segment(const Segment & seg,
                             std::map<std::string, at::Tensor> & state,
                             std::vector<at::Tensor> & u_solved_groups,
                             std::vector<at::Tensor> & g_groups) const
{
  // Per-variable predictor (optional): runs BEFORE we pack, since
  // predictor output is per-variable and lands in state[u.name].
  // Initial per-variable u defaults to zero at natural shape; the
  // predictor overrides matching names.
  _assert(!seg.givens.empty(),
          "aoti::Model: implicit segment has no givens; cannot infer batch shape.");
  auto it_g0 = state.find(seg.givens[0].name);
  _assert(it_g0 != state.end(),
          "aoti::Model: implicit segment needs given '",
          seg.givens[0].name,
          "' which is not in the state map.");
  const auto opts = it_g0->second.options();
  const auto g0 = it_g0->second;
  const auto & g0_info = seg.givens[0];
  const int64_t g0_trail =
      static_cast<int64_t>(g0_info.sub_batch_shape.size() + g0_info.base_shape.size());
  _assert(g0.dim() >= g0_trail,
          "aoti::Model: given tensor '",
          g0_info.name,
          "' ndim=",
          g0.dim(),
          " < declared sub_batch_ndim+base_ndim=",
          g0_trail);
  std::vector<int64_t> batch_shape(g0.sizes().begin(), g0.sizes().end() - g0_trail);

  auto _full_shape = [&](const Segment::VarInfo & v) -> std::vector<int64_t>
  {
    std::vector<int64_t> shape(batch_shape);
    shape.insert(shape.end(), v.sub_batch_shape.begin(), v.sub_batch_shape.end());
    shape.insert(shape.end(), v.base_shape.begin(), v.base_shape.end());
    return shape;
  };

  // Seed per-variable zero unknowns into state so the pack picks them up.
  for (const auto & v : seg.unknowns)
    state[v.name] = at::zeros(_full_shape(v), opts);

  if (seg.predictor_loader)
  {
    std::vector<at::Tensor> p_inputs;
    p_inputs.reserve(seg.predictor_inputs.size() + seg.param_inputs.size());
    for (const auto & name : seg.predictor_inputs)
    {
      auto it = state.find(name);
      _assert(it != state.end(),
              "aoti::Model: implicit segment predictor needs input '",
              name,
              "' which is not in the state map.");
      p_inputs.push_back(it->second.contiguous());
    }
    for (auto & p : _gather_params(seg.param_inputs))
      p_inputs.push_back(std::move(p));

    const auto p_outs = seg.predictor_loader->run(p_inputs);
    _assert(p_outs.size() == seg.predictor_outputs.size(),
            "aoti::Model: predictor returned ",
            p_outs.size(),
            " outputs, expected ",
            seg.predictor_outputs.size());

    // Route predictor outputs to the matching unknown slot by name.
    for (std::size_t i = 0; i < p_outs.size(); ++i)
      state[seg.predictor_outputs[i]] = p_outs[i].to(opts).contiguous();
  }

  // Pack per-group inputs at solve start (one at::cat per group).
  g_groups = _pack_groups(state, seg.given_groups);
  auto u0_groups = _pack_groups(state, seg.unknown_groups);

  // Newton solve (per-group).
  u_solved_groups = _solve_newton(seg, u0_groups, g_groups);

  // Unpack converged per-group unknowns back to per-variable state for
  // downstream forward segments / master outputs to read by name.
  _unpack_groups(u_solved_groups, seg.unknown_groups, state);
}

at::Tensor
Model::_pergroup_norm_sq(const std::vector<at::Tensor> & group_tensors,
                         const std::vector<Segment::GroupInfo> & groups)
{
  _assert(group_tensors.size() == groups.size(),
          "_pergroup_norm_sq: tensor count ",
          group_tensors.size(),
          " != group count ",
          groups.size());
  at::Tensor acc;
  for (std::size_t k = 0; k < group_tensors.size(); ++k)
  {
    const auto & t = group_tensors[k];
    const auto & g = groups[k];
    // BLOCK group tensor: (*B, *group_sub, group_base_total) -> reduce
    // sub.size()+1 trailing axes. DENSE: (*B, group_total) -> reduce 1.
    const int64_t trail =
        (g.structure == "block") ? static_cast<int64_t>(g.sub_batch_shape.size()) + 1 : 1;
    _assert(t.dim() >= trail,
            "_pergroup_norm_sq: group tensor ndim=",
            t.dim(),
            " < trail ndim=",
            trail);
    auto sq = t * t;
    std::vector<int64_t> dims;
    dims.reserve(static_cast<std::size_t>(trail));
    for (int64_t d = t.dim() - trail; d < t.dim(); ++d)
      dims.push_back(d);
    sq = sq.sum(dims);
    acc = acc.defined() ? (acc + sq) : sq;
  }
  return acc;
}

at::Tensor
Model::_pergroup_dot(const std::vector<at::Tensor> & a_groups,
                     const std::vector<at::Tensor> & b_groups,
                     const std::vector<Segment::GroupInfo> & groups)
{
  _assert(a_groups.size() == b_groups.size() && a_groups.size() == groups.size(),
          "_pergroup_dot: vector size mismatch (",
          a_groups.size(),
          "/",
          b_groups.size(),
          " vs ",
          groups.size(),
          ")");
  at::Tensor acc;
  for (std::size_t k = 0; k < a_groups.size(); ++k)
  {
    const auto & g = groups[k];
    const int64_t trail =
        (g.structure == "block") ? static_cast<int64_t>(g.sub_batch_shape.size()) + 1 : 1;
    auto prod = a_groups[k] * b_groups[k];
    std::vector<int64_t> dims;
    dims.reserve(static_cast<std::size_t>(trail));
    for (int64_t d = a_groups[k].dim() - trail; d < a_groups[k].dim(); ++d)
      dims.push_back(d);
    prod = prod.sum(dims);
    acc = acc.defined() ? (acc + prod) : prod;
  }
  return acc;
}

at::Tensor
Model::_alpha_for_group(const at::Tensor & alpha, const Segment::GroupInfo & g)
{
  // BLOCK group tensor: (*B, *group_sub, group_base_total) -> append
  // sub.size()+1 trailing size-1 axes. DENSE: append 1.
  auto out = alpha;
  const int64_t trail =
      (g.structure == "block") ? static_cast<int64_t>(g.sub_batch_shape.size()) + 1 : 1;
  for (int64_t k = 0; k < trail; ++k)
    out = out.unsqueeze(-1);
  return out;
}

std::vector<at::Tensor>
Model::_solve_newton(const Segment & seg,
                     const std::vector<at::Tensor> & u0_groups,
                     const std::vector<at::Tensor> & g_groups) const
{
  _assert(u0_groups.size() == seg.unknown_groups.size(),
          "_solve_newton: u0 group count ",
          u0_groups.size(),
          " != unknown_groups ",
          seg.unknown_groups.size());
  _assert(g_groups.size() == seg.given_groups.size(),
          "_solve_newton: g group count ",
          g_groups.size(),
          " != given_groups ",
          seg.given_groups.size());

  // Per-group u, kept at natural per-group shape (BLOCK: (*B, *group_sub,
  // group_base_total); DENSE: (*B, group_total)). The Newton loop
  // updates each group independently via `u_groups[i] = u_groups[i] +
  // alpha * du_groups[i]`, where alpha is broadcast-reshaped per
  // group's trailing axes by `_alpha_for_group`.
  std::vector<at::Tensor> u;
  u.reserve(u0_groups.size());
  for (const auto & t : u0_groups)
    u.push_back(t.contiguous());

  // Given groups stay constant throughout the solve.
  std::vector<at::Tensor> g;
  g.reserve(g_groups.size());
  for (const auto & t : g_groups)
    g.push_back(t.contiguous());

  const auto pre_tail = _gather_params(seg.param_inputs);

  // Build a loader-call input list: (*u_groups, *g_groups, *params).
  auto build_inputs = [&](const std::vector<at::Tensor> & u_pack)
  {
    std::vector<at::Tensor> v;
    v.reserve(u_pack.size() + g.size() + pre_tail.size());
    for (const auto & t : u_pack)
      v.push_back(t.contiguous());
    for (const auto & t : g)
      v.push_back(t);
    for (const auto & t : pre_tail)
      v.push_back(t);
    return v;
  };

  // Initial per-residual-group b vector.
  auto b_outs = seg.rhs_loader->run(build_inputs(u));
  _assert(b_outs.size() == seg.residual_groups.size(),
          "rhs.pt2 must return ",
          seg.residual_groups.size(),
          " tensors (one per residual group), got ",
          b_outs.size());
  // Per-batch norm of the residual: sqrt of sum across residual groups
  // of sum over (group_sub + group_base) of group^2. Leading axes are
  // the dynamic-batch shape.
  auto b0_norm_sq = _pergroup_norm_sq(b_outs, seg.residual_groups);
  auto b0_norm = b0_norm_sq.sqrt();

  const char * trace_env = std::getenv("NEML2_AOTI_TRACE_NEWTON");
  const int trace_level = trace_env ? std::atoi(trace_env) : 0;
  const bool trace = trace_level >= 1;
  const bool verbose = trace_level >= 2;

  if (check_converged(b0_norm, b0_norm, seg.atol, seg.rtol))
  {
    if (trace)
      std::cerr << "[aoti newton]iters=0 (converged at predictor) "
                << "b0_norm=" << b0_norm.max().item<double>() << std::endl;
    return u;
  }
  std::size_t reached = seg.miters;
  if (verbose)
    std::cerr << "ITERATION " << std::setw(3) << 0 << ", |R| = " << std::scientific
              << b0_norm.max().item<double>() << ", |R0| = " << std::scientific
              << b0_norm.max().item<double>() << std::endl;
  for (std::size_t i = 1; i < seg.miters; ++i)
  {
    // step graph returns (*du_groups, *b_curr_groups) of length n_u + n_r.
    // We already have b_outs from the prior iteration, so b_curr is
    // discarded (kept by the graph signature for symmetry / future use).
    const auto step_outs = seg.step_loader->run(build_inputs(u));
    _assert(step_outs.size() == seg.unknown_groups.size() + seg.residual_groups.size(),
            "step.pt2 must return n_unknown_groups + n_residual_groups tensors, got ",
            step_outs.size(),
            " (expected ",
            seg.unknown_groups.size() + seg.residual_groups.size(),
            ")");
    std::vector<at::Tensor> du(step_outs.begin(), step_outs.begin() + seg.unknown_groups.size());

    auto alpha = at::ones_like(b0_norm);
    // Armijo `b · du`: dot product across all residual+unknown groups.
    // For a square system, residual groups align with unknown groups by
    // position (residual_groups[k] is the residual block for
    // unknown_groups[k]), with matching per-group structure so trailing
    // axes match per pair.
    const auto b_dot_du = _pergroup_dot(b_outs, du, seg.unknown_groups);
    const auto nb_curr_sq = _pergroup_norm_sq(b_outs, seg.residual_groups);

    std::vector<at::Tensor> u_trial(seg.unknown_groups.size());
    std::vector<at::Tensor> b_trial;
    if (seg.ls_max_iters <= 1)
    {
      for (std::size_t k = 0; k < seg.unknown_groups.size(); ++k)
        u_trial[k] = (u[k] + du[k]).contiguous();
      b_trial = seg.rhs_loader->run(build_inputs(u_trial));
      _assert(b_trial.size() == seg.residual_groups.size(),
              "rhs.pt2 must return ",
              seg.residual_groups.size(),
              " tensors (one per residual group)");
    }
    else
    {
      for (std::size_t k_ls = 1; k_ls < seg.ls_max_iters; ++k_ls)
      {
        for (std::size_t k = 0; k < seg.unknown_groups.size(); ++k)
        {
          const auto alpha_b = _alpha_for_group(alpha, seg.unknown_groups[k]);
          u_trial[k] = (u[k] + alpha_b * du[k]).contiguous();
        }
        b_trial = seg.rhs_loader->run(build_inputs(u_trial));
        _assert(b_trial.size() == seg.residual_groups.size(),
                "rhs.pt2 must return ",
                seg.residual_groups.size(),
                " tensors (one per residual group)");

        const auto nb_trial_sq = _pergroup_norm_sq(b_trial, seg.residual_groups);
        at::Tensor crit;
        if (seg.ls_type == "STRONG_WOLFE")
          crit = (1.0 - seg.ls_c * alpha) * nb_curr_sq;
        else // BACKTRACKING
          crit = nb_curr_sq - 2.0 * seg.ls_c * alpha * b_dot_du;

        if (verbose)
          std::cerr << "     LS ITERATION " << std::setw(3) << k_ls
                    << ", min(alpha) = " << std::scientific << alpha.min().item<double>()
                    << ", max(||R||) = " << std::scientific
                    << nb_trial_sq.sqrt().max().item<double>()
                    << ", min(||Rc||) = " << std::scientific << crit.sqrt().min().item<double>()
                    << std::endl;

        const auto stop = at::logical_or(nb_trial_sq <= crit, nb_trial_sq <= seg.atol * seg.atol);
        if (stop.all().item<bool>())
          break;
        alpha = at::where(stop, alpha, alpha / seg.ls_cutback);
      }
    }

    u = std::move(u_trial);
    b_outs = std::move(b_trial);

    const auto b_norm_sq = _pergroup_norm_sq(b_outs, seg.residual_groups);
    const auto b_norm = b_norm_sq.sqrt();
    if (verbose)
      std::cerr << "ITERATION " << std::setw(3) << i << ", |R| = " << std::scientific
                << b_norm.max().item<double>() << ", |R0| = " << std::scientific
                << b0_norm.max().item<double>() << std::endl;
    const auto status = check_stop(b_norm, b0_norm, seg.atol, seg.rtol);
    if (status == StopStatus::Diverged)
      throw std::runtime_error("AOTI Newton diverged at iter " + std::to_string(i) +
                               " (non-finite residual). Consider tightening the predictor, "
                               "increasing max_linesearch_iterations, or reducing the time step.");
    if (status == StopStatus::Converged)
    {
      reached = i;
      if (trace)
        std::cerr << "[aoti newton]iters=" << i << " (converged) "
                  << "b0_norm=" << b0_norm.max().item<double>()
                  << " b_norm=" << b_norm.max().item<double>() << std::endl;
      return u;
    }
  }

  if (trace)
  {
    const auto final_norm = _pergroup_norm_sq(b_outs, seg.residual_groups).sqrt();
    std::cerr << "[aoti newton]iters=" << reached
              << " (MAXITERS HIT) b0_norm=" << b0_norm.max().item<double>()
              << " b_norm=" << final_norm.max().item<double>() << std::endl;
  }

  // Maxed out without converging. Match the C++ Newton solver convention:
  // return the last iterate rather than throwing so callers can inspect it.
  return u;
}

// ---------------------------------------------------------------------------
// Jacobian path
// ---------------------------------------------------------------------------

void
Model::_init_dstate(const at::TensorOptions & options,
                    at::IntArrayRef batch_shape,
                    std::map<std::string, at::Tensor> & dstate) const
{
  std::vector<int64_t> shape_vec(batch_shape.begin(), batch_shape.end());
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto s_k = _input_sizes[k];
    const auto c_k = _input_offsets[k];
    shape_vec.push_back(s_k);
    shape_vec.push_back(_input_total_size);
    auto block = at::zeros(shape_vec, options);
    block.narrow(/*dim=*/-1, /*start=*/c_k, /*length=*/s_k).copy_(at::eye(s_k, options));
    shape_vec.pop_back();
    shape_vec.pop_back();
    dstate[_input_names[k]] = block;
  }
}

void
Model::_run_forward_segment_jacobian(const Segment & seg,
                                     std::map<std::string, at::Tensor> & state,
                                     std::map<std::string, at::Tensor> & dstate) const
{
  std::vector<at::Tensor> inputs;
  inputs.reserve(seg.fwd_inputs.size() + seg.param_inputs.size());
  for (const auto & name : seg.fwd_inputs)
  {
    auto it = state.find(name);
    _assert(it != state.end(),
            "aoti::Model: forward segment (JVP path) needs input '",
            name,
            "' which is not in the state map.");
    inputs.push_back(it->second.contiguous());
  }
  for (auto & p : _gather_params(seg.param_inputs))
    inputs.push_back(std::move(p));

  // JVP loader returns (*outputs..., *J_pairs) where J_pairs are one
  // per (out_var, in_var) pair, row-major (outputs outer, structural
  // inputs inner) -- matches seg.jacobian_pairs.
  const auto jvp_outs = seg.jvp_loader->run(inputs);
  const std::size_t n_outs = seg.fwd_outputs.size();
  const std::size_t n_pairs = seg.jacobian_pairs.size();
  _assert(jvp_outs.size() == n_outs + n_pairs,
          "aoti::Model: JVP loader returned ",
          jvp_outs.size(),
          " tensors, expected ",
          n_outs + n_pairs,
          " (",
          n_outs,
          " outputs + ",
          n_pairs,
          " Jacobian pairs).");
  for (std::size_t i = 0; i < n_outs; ++i)
    state[seg.fwd_outputs[i]] = jvp_outs[i];

  // Master_ndim from any dstate entry; all dstate tensors share their
  // trailing M dim.
  _assert(!seg.fwd_inputs.empty(),
          "aoti::Model: forward-Jacobian segment has no fwd_inputs; cannot infer master_ndim.");
  auto first_din_it = dstate.find(seg.fwd_inputs[0]);
  _assert(first_din_it != dstate.end(),
          "aoti::Model: dstate missing input '",
          seg.fwd_inputs[0],
          "' at Jacobian composition time.");
  const int64_t M = first_din_it->second.size(-1);

  // Per-output accumulator at (*B, out_var_size, M). Batch shape from
  // dstate[fwd_inputs[0]] (which is the canonical (*B, in_var_size, M)).
  std::vector<int64_t> batch_shape;
  {
    const auto & ref = first_din_it->second;
    for (int64_t d = 0; d < ref.dim() - 2; ++d)
      batch_shape.push_back(ref.size(d));
  }
  std::map<std::string, at::Tensor> dout_acc;
  // Cache per-output var_size from the JVP value tensor (its trailing
  // shape matches the output's natural (*sub, *base)).
  for (std::size_t i = 0; i < n_outs; ++i)
  {
    const auto & val = jvp_outs[i];
    int64_t out_var_size = 1;
    for (int64_t d = static_cast<int64_t>(batch_shape.size()); d < val.dim(); ++d)
      out_var_size *= val.size(d);
    std::vector<int64_t> shape = batch_shape;
    shape.push_back(out_var_size);
    shape.push_back(M);
    dout_acc[seg.fwd_outputs[i]] = at::zeros(shape, val.options());
  }

  // Iterate pairs; per-pair matmul + reshape + accumulate.
  for (std::size_t k = 0; k < n_pairs; ++k)
  {
    const auto & pair = jvp_outs[n_outs + k];
    const auto & pinfo = seg.jacobian_pairs[k];

    int64_t in_base_total = 1;
    for (auto s : pinfo.in_base_shape)
      in_base_total *= s;
    int64_t in_sub_total = 1;
    for (auto s : pinfo.in_sub_batch_shape)
      in_sub_total *= s;
    int64_t out_base_total = 1;
    for (auto s : pinfo.out_base_shape)
      out_base_total *= s;
    const int64_t in_var_size = in_sub_total * in_base_total;

    // dstate[in_var] shape: (*B, in_var_size, M). View as (*B, *in_sub,
    // *in_base, M) -- last 1+in_base_ndim dims index over the in side.
    auto din_it = dstate.find(pinfo.in_var);
    _assert(din_it != dstate.end(),
            "aoti::Model: forward-Jacobian needs dstate['",
            pinfo.in_var,
            "'] which is missing.");
    const auto & din = din_it->second;
    std::vector<int64_t> din_target = batch_shape;
    for (auto s : pinfo.in_sub_batch_shape)
      din_target.push_back(s);
    for (auto s : pinfo.in_base_shape)
      din_target.push_back(s);
    din_target.push_back(M);
    auto din_view = din.reshape(din_target);

    // pair shape: (*B, *sub, *out_base, *in_base). View as (*B, *sub,
    // out_base_total, in_base_total) for the matmul, where *sub may be
    // empty (in_sub absent) or in_sub (chain-rule preserved). For
    // simplicity, flatten the pair's trailing 2*ndim into (out_base,
    // in_base) and rely on the matmul to broadcast against din_view.
    //
    // Tensordot over in_base axes. Use reshape to (..., out_base,
    // in_base) then matmul against (..., in_base, M).
    const int64_t pair_trail =
        static_cast<int64_t>(pinfo.out_base_shape.size() + pinfo.in_base_shape.size());
    _assert(pair.dim() >= pair_trail,
            "aoti::Model: Jacobian pair tensor ndim=",
            pair.dim(),
            " < expected trail=",
            pair_trail);
    const int64_t pair_lead = pair.dim() - pair_trail;
    std::vector<int64_t> pair_shape;
    pair_shape.reserve(static_cast<std::size_t>(pair_lead + 2));
    for (int64_t d = 0; d < pair_lead; ++d)
      pair_shape.push_back(pair.size(d));
    pair_shape.push_back(out_base_total);
    pair_shape.push_back(in_base_total);
    auto pair_view = pair.reshape(pair_shape);

    // din_view's trailing layout: (..., in_base_total, M). Reshape to
    // collapse in_base axes and (if present) in_sub axes.
    std::vector<int64_t> din_collapsed_shape = batch_shape;
    if (!pinfo.in_sub_batch_shape.empty())
    {
      for (auto s : pinfo.in_sub_batch_shape)
        din_collapsed_shape.push_back(s);
    }
    din_collapsed_shape.push_back(in_base_total);
    din_collapsed_shape.push_back(M);
    auto din_collapsed = din.reshape(din_collapsed_shape);

    // pair_view @ din_collapsed: pair_view (*B, *sub_or_none,
    // out_base_total, in_base_total) and din_collapsed
    // (*B, *in_sub, in_base_total, M). For paired BLOCK pairs (sub
    // present on both), broadcast aligns the sub. For DENSE in_var
    // (no in_sub), din shape (*B, in_base_total, M).
    auto contrib = at::matmul(pair_view, din_collapsed);
    // contrib shape (*B, *sub_union, out_base_total, M); for the DENSE
    // out_var case, accumulate as-is into (*B, out_var_size, M).
    // For BLOCK out_var (out_sub present), the sub axes need to be
    // folded into out_var_size. The current implementation supports
    // only DENSE out_var; reject BLOCK out_var here pending the future
    // BLOCK-output forward-Jacobian path.
    const int64_t accumulator_var_size = dout_acc[pinfo.out_var].size(-2);
    int64_t contrib_var_size = 1;
    for (int64_t d = static_cast<int64_t>(batch_shape.size()); d < contrib.dim() - 1; ++d)
      contrib_var_size *= contrib.size(d);
    if (contrib_var_size == accumulator_var_size)
    {
      std::vector<int64_t> flat_shape = batch_shape;
      flat_shape.push_back(accumulator_var_size);
      flat_shape.push_back(M);
      dout_acc[pinfo.out_var] = dout_acc[pinfo.out_var] + contrib.reshape(flat_shape);
    }
    else
    {
      _assert(false,
              "aoti::Model: forward-Jacobian per-pair contribution for (",
              pinfo.out_var,
              ", ",
              pinfo.in_var,
              ") has var_size ",
              contrib_var_size,
              " which does not match accumulator ",
              accumulator_var_size,
              ". BLOCK-output forward-Jacobian path is not yet implemented.");
    }
    (void)in_var_size;
  }

  for (const auto & oname : seg.fwd_outputs)
    dstate[oname] = dout_acc[oname].contiguous();
}

void
Model::_run_implicit_segment_jacobian(const Segment & seg,
                                      const std::vector<at::Tensor> & u_solved_groups,
                                      const std::vector<at::Tensor> & g_groups,
                                      std::map<std::string, at::Tensor> & dstate) const
{
  // IFT loader signature: (*u_groups, *g_groups, [params...]) -> *cells.
  // Each cell is one (row_group, col_group) entry of -du_dg in row-major
  // order (matches seg.ift_cells).
  std::vector<at::Tensor> ift_in;
  ift_in.reserve(u_solved_groups.size() + g_groups.size() + seg.param_inputs.size());
  for (const auto & t : u_solved_groups)
    ift_in.push_back(t.contiguous());
  for (const auto & t : g_groups)
    ift_in.push_back(t.contiguous());
  for (auto & p : _gather_params(seg.param_inputs))
    ift_in.push_back(std::move(p));

  const auto ift_outs = seg.ift_loader->run(ift_in);
  _assert(ift_outs.size() == seg.ift_cells.size(),
          "aoti::Model: IFT loader returned ",
          ift_outs.size(),
          " tensors, expected ",
          seg.ift_cells.size(),
          " (one per cell).");

  // Master_ndim from dstate of any given var (all dstate entries share
  // the trailing M dim).
  _assert(!seg.givens.empty(),
          "aoti::Model: implicit-segment IFT has no givens; cannot infer master_ndim.");
  auto first_dg_it = dstate.find(seg.givens[0].name);
  _assert(first_dg_it != dstate.end(),
          "aoti::Model: dstate missing given '",
          seg.givens[0].name,
          "' at IFT composition time.");
  const int64_t M = first_dg_it->second.size(-1);

  // Batch shape (leading dims of any cell's tensor, before its sub +
  // base axes). Pick cell 0 to source it.
  const auto & cell0 = ift_outs[0];
  const auto & cinfo0 = seg.ift_cells[0];
  const int64_t cell0_trail =
      static_cast<int64_t>(cinfo0.sub_batch_shape.size()) + 2; // sub + 2 base
  _assert(cell0.dim() >= cell0_trail,
          "aoti::Model: IFT cell tensor ndim=",
          cell0.dim(),
          " < expected trail=",
          cell0_trail);
  const int64_t batch_ndim = cell0.dim() - cell0_trail;
  std::vector<int64_t> batch_shape;
  batch_shape.reserve(static_cast<std::size_t>(batch_ndim));
  for (int64_t d = 0; d < batch_ndim; ++d)
    batch_shape.push_back(cell0.size(d));

  // Per-unknown accumulator at the converged-segment shape:
  // (*B, u.var_size, M). Mirrors what the old flat-matmul produced so
  // downstream forward composition sees an identical layout.
  std::map<std::string, at::Tensor> du_acc;
  for (const auto & u : seg.unknowns)
  {
    std::vector<int64_t> shape = batch_shape;
    shape.push_back(u.var_size);
    shape.push_back(M);
    du_acc[u.name] = at::zeros(shape, cell0.options());
  }

  // Iterate cells; for each, slice per-(rvar, cvar) sub-cells on the
  // last 2 dims by base offsets, dispatch matmul kind by
  // (row_structure, col_structure), accumulate into du_acc[rvar].
  for (std::size_t k = 0; k < ift_outs.size(); ++k)
  {
    const auto & cell_t = ift_outs[k];
    const auto & cinfo = seg.ift_cells[k];
    const bool row_block = (cinfo.row_structure == "block");
    const bool col_block = (cinfo.col_structure == "block");

    int64_t r_off = 0;
    for (const auto & rv : cinfo.row_vars)
    {
      int64_t r_base = 1;
      for (auto s : rv.base_shape)
        r_base *= s;
      int64_t r_sub_total = 1;
      for (auto s : rv.sub_batch_shape)
        r_sub_total *= s;

      int64_t c_off = 0;
      for (const auto & cv : cinfo.col_vars)
      {
        int64_t c_base = 1;
        for (auto s : cv.base_shape)
          c_base *= s;
        int64_t c_sub_total = 1;
        for (auto s : cv.sub_batch_shape)
          c_sub_total *= s;

        // Slice sub-cell from cell_t (last-2 dims).
        auto sub_cell = cell_t.narrow(/*dim=*/-2, /*start=*/r_off, /*length=*/r_base)
                            .narrow(/*dim=*/-1, /*start=*/c_off, /*length=*/c_base);

        // Pick the matching dg_dmaster view for cv. dstate[cv.name]
        // has shape (*B, cv.var_size, M). For a BLOCK col group, view
        // as (*B, *cv.sub, *cv.base_flat, M) but we want (*B, sub, c_base, M)
        // with sub_total flat across cv's sub axes; for DENSE col, keep
        // (*B, cv.var_size, M).
        auto dg_it = dstate.find(cv.name);
        _assert(dg_it != dstate.end(),
                "aoti::Model: IFT consumer needs dstate['",
                cv.name,
                "'] which is missing.");
        const auto & dg = dg_it->second;

        // Contribution shape: (*B, *cell_sub_for_row, r_base, M).
        at::Tensor contribution;
        if (row_block && col_block)
        {
          // PAIRED BLOCK + BLOCK: sub axes on row and col are the same
          // (single sub axis on cell tensor). dg viewed as
          // (*B, c_sub_total, c_base, M).
          std::vector<int64_t> dg_target = batch_shape;
          for (auto s : cv.sub_batch_shape)
            dg_target.push_back(s);
          dg_target.push_back(c_base);
          dg_target.push_back(M);
          auto dg_view = dg.reshape(dg_target);
          // sub_cell shape: (*B, *cell_sub, r_base, c_base); dg_view
          // (*B, *cv.sub, c_base, M); matmul → (*B, *sub, r_base, M).
          contribution = at::matmul(sub_cell, dg_view);
        }
        else if (row_block && !col_block)
        {
          // BLOCK row, DENSE col: sub_cell (*B, *row_sub, r_base,
          // c_folded). dg (*B, c_folded, M). Broadcast dg with
          // singleton sub axes.
          std::vector<int64_t> dg_target = batch_shape;
          for (std::size_t s = 0; s < cinfo.sub_batch_shape.size(); ++s)
            dg_target.push_back(1);
          dg_target.push_back(c_base);
          dg_target.push_back(M);
          auto dg_view = dg.reshape(dg_target);
          contribution = at::matmul(sub_cell, dg_view);
        }
        else if (!row_block && col_block)
        {
          // DENSE row, BLOCK col: sub_cell (*B, *col_sub, r_folded,
          // c_base). dg (*B, *cv.sub, c_base, M). matmul → (*B,
          // *col_sub, r_folded, M); SUM over the col_sub axes (cross-
          // grain reduction into the global row).
          std::vector<int64_t> dg_target = batch_shape;
          for (auto s : cv.sub_batch_shape)
            dg_target.push_back(s);
          dg_target.push_back(c_base);
          dg_target.push_back(M);
          auto dg_view = dg.reshape(dg_target);
          auto per_col = at::matmul(sub_cell, dg_view);
          std::vector<int64_t> sum_dims;
          const int64_t col_sub_ndim = static_cast<int64_t>(cv.sub_batch_shape.size());
          for (int64_t d = per_col.dim() - 2 - col_sub_ndim; d < per_col.dim() - 2; ++d)
            sum_dims.push_back(d);
          contribution = per_col.sum(sum_dims);
        }
        else
        {
          // DENSE row, DENSE col: standard matmul.
          contribution = at::matmul(sub_cell, dg);
        }

        // Accumulate into du_acc[rv.name] at shape (*B, rv.var_size, M).
        // For BLOCK row, contribution carries row sub axes; reshape to
        // (*B, sub_total * r_base, M) so it lays out var-shaped slabs.
        at::Tensor flat_contribution;
        if (row_block)
        {
          std::vector<int64_t> target = batch_shape;
          target.push_back(r_sub_total * r_base);
          target.push_back(M);
          flat_contribution = contribution.reshape(target);
          (void)c_sub_total; // not used in this branch
        }
        else
        {
          flat_contribution = contribution;
        }

        // r_var_size = (BLOCK row: r_sub_total * r_base; DENSE row:
        // r_base). du_acc is keyed by rv.name with shape (*B,
        // rv.var_size, M).
        int64_t r_var_size = (row_block) ? (r_sub_total * r_base) : r_base;
        _assert(r_var_size == rv.var_size,
                "aoti::Model: IFT row-var size mismatch for '",
                rv.name,
                "'");
        du_acc[rv.name] = du_acc[rv.name] + flat_contribution;

        c_off += c_base;
      }
      r_off += r_base;
    }
  }

  // Write final per-unknown du into dstate. Negation: cells carry
  // -du_dg already (Python side returns -du_dg.tensors[i][j]).
  for (const auto & u : seg.unknowns)
    dstate[u.name] = du_acc[u.name].contiguous();
}
} // namespace neml2::aoti
