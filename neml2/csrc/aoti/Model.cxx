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

#include <fstream>
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

// Row-stack `dstate` blocks for the named variables. Each block is
// (*B, var_size, M); concatenation along dim -2 gives (*B, sum(var_size), M).
at::Tensor
row_stack_dstate(const std::vector<std::string> & names,
                 const std::map<std::string, at::Tensor> & dstate)
{
  std::vector<at::Tensor> parts;
  parts.reserve(names.size());
  for (const auto & name : names)
  {
    auto it = dstate.find(name);
    _assert(it != dstate.end(),
            "aoti::Model: dstate is missing entry for variable '",
            name,
            "' needed by the Jacobian composition.");
    parts.push_back(it->second);
  }
  return at::cat(parts, /*dim=*/-2).contiguous();
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
} // namespace

Model::Model(const std::filesystem::path & meta_path)
{
  std::ifstream meta_f(meta_path);
  _assert(static_cast<bool>(meta_f),
          "aoti::Model: failed to open metadata file '",
          meta_path.string(),
          "'. Path must point at the _meta.json written by `neml2-compile`.");
  const auto meta = nlohmann::json::parse(meta_f);

  // Schema-version handshake. v2 is the current format -- bakes-by-default,
  // optional `parameters` array for promoted entries. Older v1 caches lack
  // those keys and must be regenerated; we fail loudly instead of silently
  // misinterpreting them.
  static constexpr int kSupportedSchemaVersion = 2;
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

    if (seg_kind == "forward")
    {
      seg.kind = SegmentKind::Forward;
      seg.fwd_loader = make_loader(cache_dir / seg_meta["package"].get<std::string>());
      if (seg_meta.contains("jvp_package"))
        seg.jvp_loader = make_loader(cache_dir / seg_meta["jvp_package"].get<std::string>());
      for (const auto & v : seg_meta["inputs"])
      {
        seg.fwd_inputs.push_back(v["name"].get<std::string>());
        seg.fwd_input_sizes.push_back(v["var_size"].get<int>());
      }
      for (const auto & v : seg_meta["outputs"])
      {
        seg.fwd_outputs.push_back(v["name"].get<std::string>());
        seg.fwd_output_sizes.push_back(v["var_size"].get<int>());
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
      {
        seg.unknowns.push_back(v["name"].get<std::string>());
        seg.unknown_sizes.push_back(v["var_size"].get<int>());
        if (v.contains("unflattened_shape"))
          seg.unknown_unflattened_shapes.push_back(
              v["unflattened_shape"].get<std::vector<int64_t>>());
        else
          seg.unknown_unflattened_shapes.emplace_back();
      }
      for (const auto & v : seg_meta["givens"])
      {
        seg.givens.push_back(v["name"].get<std::string>());
        seg.given_sizes.push_back(v["var_size"].get<int>());
        if (v.contains("unflattened_shape"))
          seg.given_unflattened_shapes.push_back(
              v["unflattened_shape"].get<std::vector<int64_t>>());
        else
          seg.given_unflattened_shapes.emplace_back();
      }
      seg.u_size = seg_meta["u_size"].get<int>();
      seg.g_size = seg_meta["g_size"].get<int>();
      seg.atol = seg_meta["atol"].get<double>();
      seg.rtol = seg_meta["rtol"].get<double>();
      seg.miters = static_cast<std::size_t>(seg_meta["miters"].get<int>());
      _assert(seg.u_size > 0 && seg.g_size >= 0,
              "aoti::Model: implicit segment ",
              i,
              " has invalid u/g sizes (u_size=",
              seg.u_size,
              ", g_size=",
              seg.g_size,
              ").");

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
      at::Tensor u_solved;
      at::Tensor g_flat;
      _run_implicit_segment(seg, state, u_solved, g_flat);
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
      at::Tensor u_solved;
      at::Tensor g_flat;
      _run_implicit_segment(seg, state, u_solved, g_flat);
      _assert(static_cast<bool>(seg.ift_loader),
              "aoti::Model::jacobian: implicit segment is missing its ift_package loader. "
              "Re-export via `neml2-compile`.");
      _run_implicit_segment_jacobian(seg, u_solved, g_flat, dstate);
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

void
Model::_run_implicit_segment(const Segment & seg,
                             std::map<std::string, at::Tensor> & state,
                             at::Tensor & u_solved,
                             at::Tensor & g_flat) const
{
  // Pack givens into a flat g vector.
  std::vector<at::Tensor> g_parts;
  g_parts.reserve(seg.givens.size());
  for (std::size_t i = 0; i < seg.givens.size(); ++i)
  {
    auto it = state.find(seg.givens[i]);
    _assert(it != state.end(),
            "aoti::Model: implicit segment needs given '",
            seg.givens[i],
            "' which is not in the state map.");
    g_parts.push_back(
        _reshape_to_flat_slot(it->second, seg.given_sizes[i], seg.given_unflattened_shapes[i]));
  }
  g_flat = at::cat(g_parts, /*dim=*/-1).contiguous();

  // Initial guess: predictor if loaded, otherwise zero.
  const auto batch_shape = g_flat.sizes().slice(0, g_flat.dim() - 1);
  std::vector<int64_t> u0_shape(batch_shape.begin(), batch_shape.end());
  u0_shape.push_back(seg.u_size);

  at::Tensor u0;
  if (!seg.predictor_loader)
  {
    u0 = at::zeros(u0_shape, g_flat.options());
  }
  else
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

    u0 = at::zeros(u0_shape, g_flat.options());
    // Scatter predictor outputs into u0 at the matching unknown offsets.
    for (std::size_t i = 0; i < p_outs.size(); ++i)
    {
      const auto & pname = seg.predictor_outputs[i];
      int u_idx = -1;
      for (std::size_t j = 0; j < seg.unknowns.size(); ++j)
        if (seg.unknowns[j] == pname)
        {
          u_idx = static_cast<int>(j);
          break;
        }
      if (u_idx < 0)
        continue;
      int64_t offset = 0;
      for (int j = 0; j < u_idx; ++j)
        offset += seg.unknown_sizes[j];
      const int64_t sz = seg.unknown_sizes[u_idx];
      auto p = _reshape_to_flat_slot(p_outs[i], sz, seg.unknown_unflattened_shapes[u_idx]);
      u0.narrow(-1, offset, sz).copy_(p);
    }
  }

  // Newton solve (private method, folded in from the old aoti::solve_newton).
  u_solved = _solve_newton(seg, u0, g_flat);

  // Unpack solved unknowns into state.
  int64_t offset = 0;
  for (std::size_t i = 0; i < seg.unknowns.size(); ++i)
  {
    const int64_t sz = seg.unknown_sizes[i];
    auto slice = u_solved.narrow(-1, offset, sz);
    state[seg.unknowns[i]] =
        _reshape_from_flat_slot(slice, sz, seg.unknown_unflattened_shapes[i]).contiguous();
    offset += sz;
  }
}

at::Tensor
Model::_solve_newton(const Segment & seg, const at::Tensor & u0, const at::Tensor & g_flat) const
{
  _assert(u0.dim() >= 1 && u0.size(-1) == seg.u_size,
          "_solve_newton: u0 last dim must equal u_size=",
          seg.u_size,
          ", got ",
          u0.sizes());
  _assert(g_flat.dim() >= 1 && g_flat.size(-1) == seg.g_size,
          "_solve_newton: g_flat last dim must equal g_size=",
          seg.g_size,
          ", got ",
          g_flat.sizes());

  // .contiguous() is mandatory on every AOTI input.
  auto u = u0.contiguous();
  auto g = g_flat.contiguous();

  const auto pre_tail = _gather_params(seg.param_inputs);

  auto with_params = [&](const std::vector<at::Tensor> & head)
  {
    std::vector<at::Tensor> v;
    v.reserve(head.size() + pre_tail.size());
    for (const auto & t : head)
      v.push_back(t);
    for (const auto & t : pre_tail)
      v.push_back(t);
    return v;
  };

  // Initial residual / norm for the relative convergence baseline.
  const auto b0_out = seg.rhs_loader->run(with_params({u, g}));
  _assert(b0_out.size() == 1, "DenseRHS .pt2 must return one tensor");
  auto b = b0_out[0];
  auto b0_norm = b.norm(2, /*dim=*/-1);
  if (check_converged(b0_norm, b0_norm, seg.atol, seg.rtol))
    return u;

  for (std::size_t i = 1; i < seg.miters; ++i)
  {
    // Fused step: assemble + solve + update + new residual in one graph.
    const auto outs = seg.step_loader->run(with_params({u.contiguous(), g}));
    _assert(outs.size() == 2, "DenseNewtonStep .pt2 must return (u_new, b_new)");
    u = outs[0];
    b = outs[1];

    const auto b_norm = b.norm(2, /*dim=*/-1);
    if (check_converged(b_norm, b0_norm, seg.atol, seg.rtol))
      return u;
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

  // JVP loader returns (*outputs..., J).
  const auto jvp_outs = seg.jvp_loader->run(inputs);
  _assert(jvp_outs.size() == seg.fwd_outputs.size() + 1,
          "aoti::Model: JVP loader returned ",
          jvp_outs.size(),
          " tensors, expected ",
          seg.fwd_outputs.size() + 1,
          " (",
          seg.fwd_outputs.size(),
          " outputs + 1 Jacobian).");
  for (std::size_t i = 0; i < seg.fwd_outputs.size(); ++i)
    state[seg.fwd_outputs[i]] = jvp_outs[i];
  const auto & J = jvp_outs.back();

  // dseg_in: (*B, sum(in_sizes), M); dseg_out = J @ dseg_in : (*B, sum(out), M).
  const auto dseg_in = row_stack_dstate(seg.fwd_inputs, dstate);
  const auto dseg_out = at::matmul(J, dseg_in).contiguous();

  int64_t offset = 0;
  for (std::size_t i = 0; i < seg.fwd_outputs.size(); ++i)
  {
    const auto sz = seg.fwd_output_sizes[i];
    dstate[seg.fwd_outputs[i]] =
        dseg_out.narrow(/*dim=*/-2, /*start=*/offset, /*length=*/sz).contiguous();
    offset += sz;
  }
}

void
Model::_run_implicit_segment_jacobian(const Segment & seg,
                                      const at::Tensor & u_solved,
                                      const at::Tensor & g_flat,
                                      std::map<std::string, at::Tensor> & dstate) const
{
  // IFT loader signature: (u_flat, g_flat, [params...]) -> du/dg of shape
  // (*B, u_size, g_size).
  std::vector<at::Tensor> ift_in;
  ift_in.reserve(2 + seg.param_inputs.size());
  ift_in.push_back(u_solved.contiguous());
  ift_in.push_back(g_flat);
  for (auto & p : _gather_params(seg.param_inputs))
    ift_in.push_back(std::move(p));

  const auto ift_outs = seg.ift_loader->run(ift_in);
  _assert(ift_outs.size() == 1,
          "aoti::Model: IFT loader returned ",
          ift_outs.size(),
          " tensors, expected 1.");
  const auto & du_dg = ift_outs.front();

  // dg/d(master): row-stack dstate columns for each given.
  const auto dg_dmaster = row_stack_dstate(seg.givens, dstate);
  const auto du_dmaster = at::matmul(du_dg, dg_dmaster).contiguous();

  int64_t offset = 0;
  for (std::size_t i = 0; i < seg.unknowns.size(); ++i)
  {
    const auto sz = seg.unknown_sizes[i];
    dstate[seg.unknowns[i]] =
        du_dmaster.narrow(/*dim=*/-2, /*start=*/offset, /*length=*/sz).contiguous();
    offset += sz;
  }
}

// ---------------------------------------------------------------------------
// Static reshape helpers (Scalar slot edge case).
// ---------------------------------------------------------------------------

at::Tensor
Model::_reshape_to_flat_slot(const at::Tensor & t,
                             int64_t var_size,
                             const std::vector<int64_t> & unflattened_shape)
{
  if (unflattened_shape.empty())
    return t.unsqueeze(-1);
  const int64_t n_trailing = static_cast<int64_t>(unflattened_shape.size());
  _assert(t.dim() >= n_trailing,
          "aoti::Model: tensor with ndim=",
          t.dim(),
          " is missing trailing storage axes for unflattened_shape of size ",
          n_trailing);
  std::vector<int64_t> new_shape;
  new_shape.reserve(t.dim() - n_trailing + 1);
  for (int64_t d = 0; d < t.dim() - n_trailing; ++d)
    new_shape.push_back(t.size(d));
  new_shape.push_back(var_size);
  return t.reshape(new_shape);
}

at::Tensor
Model::_reshape_from_flat_slot(const at::Tensor & slice,
                               int64_t var_size,
                               const std::vector<int64_t> & unflattened_shape)
{
  if (unflattened_shape.empty())
  {
    if (var_size == 1)
      return slice.squeeze(-1);
    return slice;
  }
  std::vector<int64_t> new_shape;
  new_shape.reserve(slice.dim() - 1 + unflattened_shape.size());
  for (int64_t d = 0; d < slice.dim() - 1; ++d)
    new_shape.push_back(slice.size(d));
  for (int64_t s : unflattened_shape)
    new_shape.push_back(s);
  return slice.reshape(new_shape);
}
} // namespace neml2::aoti
