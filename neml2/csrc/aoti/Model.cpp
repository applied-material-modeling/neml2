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

  // Schema-version handshake. v5 moves implicit-segment I/O from a
  // single packed (*dyn, u_size) flat slab to per-variable tensors in
  // their natural (*dyn, *sub_batch, *base) shape; the per-segment
  // metadata records per-variable sub_batch_shape / base_shape so the
  // C++ side can pack the segment loader's positional inputs and so
  // convergence-norm bookkeeping can sum per-variable contributions.
  // Older caches must be regenerated; we fail loudly instead of
  // silently misinterpreting them.
  // dependencies: aoti.schema_version
  static constexpr int kSupportedSchemaVersion = 4;
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
          seg.jacobian_pairs.push_back(std::move(pi));
        }
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

std::vector<at::Tensor>
Model::Impl::_gather_params(const std::vector<std::string> & names) const
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

const std::vector<int> &
Model::input_sizes() const noexcept
{
  return _impl->input_sizes();
}

const std::vector<int> &
Model::output_sizes() const noexcept
{
  return _impl->output_sizes();
}

std::map<std::string, at::Tensor>
Model::forward(const std::map<std::string, at::Tensor> & inputs) const
{
  return _impl->forward(inputs);
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::jvp(const std::map<std::string, at::Tensor> & inputs,
           const std::map<std::string, at::Tensor> & tangents) const
{
  return _impl->jvp(inputs, tangents);
}

std::pair<std::map<std::string, at::Tensor>, at::Tensor>
Model::jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  return _impl->jacobian(inputs);
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
