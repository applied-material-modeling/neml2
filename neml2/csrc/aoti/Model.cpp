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

// ----------------------------------------------------------------------------
// Model construction: metadata parse + the public-facade forwarders
// ----------------------------------------------------------------------------
// Holds `Model::Impl`'s constructor (parses `_meta.json`, loads every `.pt2`
// segment) and `_gather_params`, plus the thin `Model` facade that forwards
// every public call onto the opaque Impl. The op/solve/Jacobian method bodies
// live in ops.cpp / solve.cpp / jacobian.cpp.

// internal.h (which pulls in the ATen umbrella) must precede <nlohmann/json.hpp>:
// in a debug build json's JSON_ASSERT expands to assert(), and the ATen/c10
// include chain is what brings the glibc __assert_fail declaration into scope.
#include "neml2/csrc/aoti/internal.h"

#include <fstream>

#include <nlohmann/json.hpp>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace neml2::aoti
{
namespace
{
// TU-local helpers used only by the metadata-parsing constructor below.
// Build a loader with the consistent constructor args used across all AOTI
// artifacts produced by neml2-compile. `device_index` is -1 for cpu artifacts
// and cuda artifacts that target the current device; a concrete index pins a
// cuda artifact onto a specific GPU (used by the multi-device dispatcher).
std::unique_ptr<torch::inductor::AOTIModelPackageLoader>
make_loader(const std::filesystem::path & path, int device_index = -1)
{
  return std::make_unique<torch::inductor::AOTIModelPackageLoader>(path.string(),
                                                                   /*model_name=*/"model",
                                                                   /*run_single_threaded=*/false,
                                                                   /*num_runners=*/1,
                                                                   device_index);
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
} // namespace

Model::Impl::Impl(const std::filesystem::path & meta_path,
                  std::optional<at::Device> device_override)
{
  std::ifstream meta_f(meta_path);
  _assert(static_cast<bool>(meta_f),
          "aoti::Model: failed to open metadata file '",
          meta_path.string(),
          "'. Path must point at the _meta.json written by `neml2-compile`.");
  const auto meta = nlohmann::json::parse(meta_f);

  // Schema-version handshake. v5 makes the runtime variable-native: the master
  // `inputs`/`outputs` metadata records each variable's full `base_shape`, so
  // the C++ side reports input_base_shapes()/output_base_shapes(), validates
  // canonical (*B, *base) inputs, and returns unflattened jvp (*B, *out_base)
  // and variable-pair jacobian blocks (*B, *out_base, *in_base). (v5 also keeps
  // the per-segment per-variable sub_batch_shape / base_shape used to pack the
  // segment loaders' positional inputs and sum convergence norms.) Older caches
  // lack the master base_shape; we fail loudly instead of misinterpreting them.
  // dependencies: aoti.schema_version (NOT auto-managed -- C++ `//` comments are
  // not scanned by scripts/dep_manager.py; keep this literal in sync by hand).
  static constexpr int kSupportedSchemaVersion = 7;
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

  // A device override refines the concrete device index (e.g. cuda:1) but may
  // not change the device *type* -- the AOTI graphs are compiled for cpu xor
  // cuda kernels and cannot be reinterpreted across that boundary.
  if (device_override.has_value())
  {
    _assert(device_override->type() == _device.type(),
            "aoti::Model: device_override type '",
            c10::DeviceTypeName(device_override->type()),
            "' does not match the artifact's compiled device type '",
            c10::DeviceTypeName(_device.type()),
            "' in '",
            meta_path.string(),
            "'. A cpu artifact cannot be loaded onto cuda (or vice versa); "
            "recompile for the target device.");
    _device = *device_override;
  }

  // Concrete loader index: -1 lets the loader use the current device (cpu, or
  // the ambient cuda device); a concrete cuda index pins this Model to a GPU.
  const int dev_idx =
      (_device.is_cuda() && _device.has_index()) ? static_cast<int>(_device.index()) : -1;

  // .pt2 basenames are resolved against the directory holding the metadata
  // file itself.
  const auto cache_dir = meta_path.parent_path();

  // Master inputs / outputs.
  _input_names.reserve(meta["inputs"].size());
  _input_base_shapes.reserve(meta["inputs"].size());
  _input_sizes.reserve(meta["inputs"].size());
  _input_offsets.reserve(meta["inputs"].size());
  for (const auto & v : meta["inputs"])
  {
    const auto name = v["name"].get<std::string>();
    const auto sz = v["var_size"].get<int>();
    _input_names.push_back(name);
    _input_base_shapes.push_back(v["base_shape"].get<std::vector<int64_t>>());
    _input_sub_batch_shapes.push_back(v.contains("sub_batch_shape")
                                          ? v["sub_batch_shape"].get<std::vector<int64_t>>()
                                          : std::vector<int64_t>{});
    _input_offsets.push_back(_input_total_size);
    _input_sizes.push_back(sz);
    _input_total_size += sz;
  }
  _output_names.reserve(meta["outputs"].size());
  _output_base_shapes.reserve(meta["outputs"].size());
  _output_sizes.reserve(meta["outputs"].size());
  for (const auto & v : meta["outputs"])
  {
    _output_names.push_back(v["name"].get<std::string>());
    _output_base_shapes.push_back(v["base_shape"].get<std::vector<int64_t>>());
    _output_sizes.push_back(v["var_size"].get<int>());
  }

  // Master (out, in) derivative pairs the artifact supports (v6+). Absent /
  // empty => no derivative graphs compiled; jvp() / jacobian() raise. Kept in
  // the metadata's (output-order, input-order) so the public maps iterate
  // rows/cols consistently.
  if (meta.contains("derivatives"))
    for (const auto & pr : meta["derivatives"])
    {
      auto o = pr.at(0).get<std::string>();
      auto i = pr.at(1).get<std::string>();
      _deriv_by_out[o].push_back(i);
      _derivatives.emplace_back(std::move(o), std::move(i));
    }

  // Master (out, param) parameter-derivative pairs the artifact supports (v7+).
  // Absent / empty => no parameter-Jacobian graph; param_jacobian() / param_vjp()
  // raise. Separate from `_derivatives` (which is over structural inputs).
  if (meta.contains("parameter_derivatives"))
    for (const auto & pr : meta["parameter_derivatives"])
    {
      auto o = pr.at(0).get<std::string>();
      auto p = pr.at(1).get<std::string>();
      _param_derivatives.emplace_back(std::move(o), std::move(p));
    }

  // Narrowed "dense auxiliary B matrix" layout: the distinct requested input
  // directions, in `_input_names` order, with their column offsets in the
  // narrowed carrier. The multi-segment dense Jacobian seeds + composes over
  // only these columns instead of every master input.
  {
    std::map<std::string, std::size_t> in_idx;
    for (std::size_t j = 0; j < _input_names.size(); ++j)
      in_idx[_input_names[j]] = j;
    std::vector<char> seen(_input_names.size(), 0);
    for (const auto & [o, i] : _derivatives)
    {
      const auto j = in_idx.at(i);
      if (!seen[j])
      {
        seen[j] = 1;
        _req_input_offset[i] = _req_total_size;
        _req_input_size[i] = _input_sizes[j];
        _req_inputs.push_back(i);
        _req_total_size += _input_sizes[j];
      }
    }
  }

  // Promoted parameters (v2; empty in the common fully-baked case).
  if (meta.contains("parameters"))
  {
    for (const auto & p : meta["parameters"])
    {
      const auto name = p["name"].get<std::string>();
      const auto dtype = parse_dtype(p.value("dtype", std::string("float64")));
      const auto shape = p["shape"].get<std::vector<int64_t>>();
      // Natural base shape (v7). Fallback to the full stored shape (== base for an
      // unbatched parameter) keeps older single-param paths robust.
      _param_base_shapes[name] = p.value("param_base_shape", shape);
      auto device = p.contains("device") ? parse_device(p["device"].get<std::string>()) : _device;
      // Promoted params must land on the same concrete GPU as the graphs: when
      // the model device carries an index (override-aware), inherit it.
      if (device.is_cuda() && _device.is_cuda() && _device.has_index())
        device.set_index(_device.index());
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

  // Narrowed parameter-column layout for the multi-segment parameter-Jacobian
  // carrier: the distinct requested promoted parameters (in _param_derivatives
  // order) with their column offsets / sizes. The parameter size is the stored
  // parameter's element count (natural base shape -- scalar => 1). Built after
  // _named_parameters is populated. Parallel to the _req_input_* block above.
  for (const auto & [o, p] : _param_derivatives)
  {
    if (_req_param_offset.count(p))
      continue; // already laid out (a param may pair with several outputs)
    auto pit = _named_parameters.find(p);
    _assert(pit != _named_parameters.end(),
            "aoti::Model: parameter-derivative references promoted parameter '",
            p,
            "' which is missing from named_parameters().");
    // The carrier column band is PER-BASE-ELEMENT (prod of the natural base), not
    // the stored numel -- a batched parameter `(*pbatch, *base)` still contributes
    // only `prod(base)` columns (its batch aligns with the carrier's batch).
    int64_t psize = 1;
    for (const auto s : _param_base_shapes.at(p))
      psize *= s;
    _req_param_offset[p] = _param_total_size;
    _req_param_size[p] = psize;
    _req_params.push_back(p);
    _param_total_size += psize;
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
      seg.fwd_loader = make_loader(cache_dir / seg_meta["package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("jvp_package"))
        seg.jvp_loader =
            make_loader(cache_dir / seg_meta["jvp_package"].get<std::string>(), dev_idx);
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
          pi.batch_independent = p.value("batch_independent", false);
          seg.jacobian_pairs.push_back(std::move(pi));
        }
      }
      // v7 parameter-derivative graphs (present iff parameter derivatives were
      // requested). The param-Jacobian loader returns one dense block per
      // (out, param) pair (ONLY blocks, no value outputs); the param-VJP loader
      // returns one gradient per parameter given output cotangents.
      if (seg_meta.contains("param_jacobian_package"))
      {
        seg.param_jacobian_loader =
            make_loader(cache_dir / seg_meta["param_jacobian_package"].get<std::string>(), dev_idx);
        for (const auto & p : seg_meta["param_jacobian_pairs"])
        {
          Segment::ParamPairInfo pi;
          pi.out_var = p["out_var"].get<std::string>();
          pi.param = p["param"].get<std::string>();
          if (p.contains("out_base_shape"))
            pi.out_base_shape = p["out_base_shape"].get<std::vector<int64_t>>();
          if (p.contains("param_base_shape"))
            pi.param_base_shape = p["param_base_shape"].get<std::vector<int64_t>>();
          seg.param_jacobian_pairs.push_back(std::move(pi));
        }
      }
      if (seg_meta.contains("param_vjp_package"))
      {
        seg.param_vjp_loader =
            make_loader(cache_dir / seg_meta["param_vjp_package"].get<std::string>(), dev_idx);
        seg.param_vjp_params = seg_meta["param_vjp_params"].get<std::vector<std::string>>();
        seg.param_vjp_outputs = seg_meta["param_vjp_outputs"].get<std::vector<std::string>>();
      }
    }
    else if (seg_kind == "implicit")
    {
      seg.kind = SegmentKind::Implicit;
      seg.rhs_loader = make_loader(cache_dir / seg_meta["rhs_package"].get<std::string>(), dev_idx);
      seg.step_loader =
          make_loader(cache_dir / seg_meta["step_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("ift_package"))
        seg.ift_loader =
            make_loader(cache_dir / seg_meta["ift_package"].get<std::string>(), dev_idx);

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

      // Per-(unknown, given) IFT Jacobian-pair metadata (v6). The IFT loader
      // emits one block per pair (via AssembledMatrix.disassemble), in
      // jacobian_pairs order, so the runtime composes them against dg_dmaster
      // exactly like a forward segment's per-pair blocks.
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
          pi.batch_independent = p.value("batch_independent", false);
          seg.jacobian_pairs.push_back(std::move(pi));
        }
      }
      // v8 implicit parameter-derivative graph (ParamIFT). Present iff a
      // parameter derivative targeting a promoted parameter inside the residual
      // was requested. Emits one dense du/dθ block per (unknown, param) pair in
      // param_jacobian_pairs order (unknowns outer, params inner).
      if (seg_meta.contains("param_ift_package"))
      {
        seg.param_ift_loader =
            make_loader(cache_dir / seg_meta["param_ift_package"].get<std::string>(), dev_idx);
        for (const auto & p : seg_meta["param_jacobian_pairs"])
        {
          Segment::ParamPairInfo pi;
          pi.out_var = p["out_var"].get<std::string>();
          pi.param = p["param"].get<std::string>();
          if (p.contains("out_base_shape"))
            pi.out_base_shape = p["out_base_shape"].get<std::vector<int64_t>>();
          if (p.contains("param_base_shape"))
            pi.param_base_shape = p["param_base_shape"].get<std::vector<int64_t>>();
          seg.param_jacobian_pairs.push_back(std::move(pi));
        }
      }
      // Solver convergence / line-search configuration is no longer baked into
      // the metadata (schema v4). It is supplied at load time via
      // Model::set_solver_config (driven from the stub's [Solvers] block).
      _assert(!seg.unknowns.empty(),
              "aoti::Model: implicit segment ",
              i,
              " has zero unknowns -- an implicit system needs at least one "
              "unknown to solve for.");

      if (seg_meta.contains("predictor_package"))
      {
        seg.predictor_loader =
            make_loader(cache_dir / seg_meta["predictor_package"].get<std::string>(), dev_idx);
        for (const auto & v : seg_meta["predictor_inputs"])
          seg.predictor_inputs.push_back(v["name"].get<std::string>());
        for (const auto & v : seg_meta["predictor_outputs"])
          seg.predictor_outputs.push_back(v["name"].get<std::string>());
        // The predictor is compiled without the residual's promoted tail, so it
        // currently takes no promoted params (the field stays empty unless a
        // future export threads them in explicitly).
        if (seg_meta.contains("predictor_param_inputs"))
          seg.predictor_param_inputs =
              seg_meta["predictor_param_inputs"].get<std::vector<std::string>>();
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

const at::Tensor &
Model::Impl::_resolve_param(const std::string & name) const
{
  if (_param_overrides != nullptr)
  {
    auto oit = _param_overrides->find(name);
    if (oit != _param_overrides->end())
      return oit->second;
  }
  auto it = _named_parameters.find(name);
  _assert(it != _named_parameters.end(),
          "aoti::Model: segment references promoted parameter '",
          name,
          "' which is missing from named_parameters(). Either the user mutated the map "
          "or the metadata is inconsistent.");
  return it->second;
}

std::vector<at::Tensor>
Model::Impl::_gather_params(const std::vector<std::string> & names) const
{
  std::vector<at::Tensor> out;
  out.reserve(names.size());
  for (const auto & n : names)
    out.push_back(_resolve_param(n).contiguous());
  return out;
}

// ----------------------------------------------------------------------------
// Public facade: forward every call onto the opaque Impl.
// ----------------------------------------------------------------------------

Model::Model(const std::filesystem::path & meta_path, std::optional<at::Device> device_override)
  : _impl(std::make_unique<Impl>(meta_path, device_override))
{
}

Model::~Model() = default;

const std::vector<std::string> &
Model::input_names() const noexcept
{
  return _impl->input_names();
}

const std::vector<std::string> &
Model::output_names() const noexcept
{
  return _impl->output_names();
}

const std::vector<std::vector<int64_t>> &
Model::input_base_shapes() const noexcept
{
  return _impl->input_base_shapes();
}

const std::vector<std::vector<int64_t>> &
Model::output_base_shapes() const noexcept
{
  return _impl->output_base_shapes();
}

// The three ops run through `_guarded` so every exception leaving the public
// surface is a neml2 Exception with a meaningful `recoverable()`: a recoverable
// ConvergenceError from the Newton solve passes through, while a foreign torch
// error (e.g. a shape / device mismatch raised inside a compiled graph) is
// normalized to a non-recoverable FatalError.
std::map<std::string, at::Tensor>
Model::forward(const std::map<std::string, at::Tensor> & inputs,
               const std::map<std::string, at::Tensor> & param_overrides) const
{
  return _guarded([&] { return _impl->forward(inputs, param_overrides); });
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::jvp(const std::map<std::string, at::Tensor> & inputs,
           const std::map<std::string, at::Tensor> & tangents,
           const std::map<std::string, at::Tensor> & param_overrides) const
{
  return _guarded([&] { return _impl->jvp(inputs, tangents, param_overrides); });
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::jacobian(const std::map<std::string, at::Tensor> & inputs,
                const std::map<std::string, at::Tensor> & param_overrides) const
{
  return _guarded([&] { return _impl->jacobian(inputs, param_overrides); });
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::param_jacobian(const std::map<std::string, at::Tensor> & inputs,
                      const std::map<std::string, at::Tensor> & param_overrides) const
{
  return _guarded([&] { return _impl->param_jacobian(inputs, param_overrides); });
}

std::map<std::string, at::Tensor>
Model::param_vjp(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & cotangents,
                 const std::map<std::string, at::Tensor> & param_overrides) const
{
  return _guarded([&] { return _impl->param_vjp(inputs, cotangents, param_overrides); });
}

std::map<std::string, at::Tensor> &
Model::named_parameters() noexcept
{
  return _impl->named_parameters();
}

const std::map<std::string, at::Tensor> &
Model::named_parameters() const noexcept
{
  return _impl->named_parameters();
}

const std::map<std::string, std::vector<int64_t>> &
Model::parameter_base_shapes() const noexcept
{
  return _impl->parameter_base_shapes();
}

void
Model::set_parameter(const std::string & name, const at::Tensor & value)
{
  auto & params = _impl->named_parameters();
  auto it = params.find(name);
  _assert(it != params.end(),
          "set_parameter: '",
          name,
          "' is not a promoted parameter; only entries that appear in "
          "named_parameters() may be set.");
  it->second = value.contiguous();
}

at::Device
Model::device() const noexcept
{
  return _impl->device();
}

at::ScalarType
Model::dtype() const noexcept
{
  return _impl->dtype();
}

void
Model::set_solver_config(const SolverConfig & config)
{
  _impl->_solver_config = config;
}

} // namespace neml2::aoti
