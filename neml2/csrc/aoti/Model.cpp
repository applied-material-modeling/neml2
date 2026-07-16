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
// Holds `Model::Impl`'s constructor (parses `metadata.json`, loads every `.pt2`
// segment) and `_gather_params`, plus the thin `Model` facade that forwards
// every public call onto the opaque Impl. The op/solve/Jacobian method bodies
// live in ops.cpp / solve.cpp / jacobian.cpp.

// internal.h (which pulls in the ATen umbrella) must precede <nlohmann/json.hpp>:
// in a debug build json's JSON_ASSERT expands to assert(), and the ATen/c10
// include chain is what brings the glibc __assert_fail declaration into scope.
#include "neml2/csrc/aoti/internal.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>

#include <nlohmann/json.hpp>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace neml2::aoti
{
namespace
{
// TU-local helpers used only by the metadata-parsing constructor below.

#ifdef _WIN32
// Windows-only AOTI extraction-temp isolation.
//
// torch's `AOTIModelPackageLoader` extracts a `.pt2` archive straight into
// `std::filesystem::temp_directory_path()` on Windows -- with NO unique
// per-load subdirectory (unlike its POSIX path, which `mkdtemp()`s a fresh
// `/tmp` dir per load) -- and `recursive_rmdir`s that same directory in its
// destructor. Two loads of a byte-identical compiled segment (e.g. two model
// variants that differ only in runtime solver config, whose `model_residual`
// `.pt2` is identical) therefore resolve to the SAME
// `%TEMP%\<pkg>\data\aotinductor\model\<hash>.wrapper.pyd`; the first load
// keeps that `.pyd` (a loaded DLL) locked for the process lifetime, so the
// second extraction fails with a Windows sharing violation ("error code: 32
// ... file open failed").
//
// Give every loader its own unique temp dir for the duration of construction
// (torch reads `%TMP%`/`%TEMP%` via `GetTempPath`), then restore the prior
// environment. torch captures the now-unique dir as its `temp_dir_` and cleans
// up only that subdir on destruction. Serialized by a mutex because the
// override mutates process-global environment state: neml2 itself constructs
// loaders serially, but a host embedding the runtime may not.
//
// POSIX needs none of this -- torch `mkdtemp()`s a unique `/tmp` dir per load
// there (and ignores `TMPDIR`), so there is no shared extraction root to
// collide on.
class ScopedExtractTempDir
{
public:
  ScopedExtractTempDir()
  {
    static std::atomic<std::uint64_t> counter{0};
    _save("TMP", _old_tmp, _had_tmp);
    _save("TEMP", _old_temp, _had_temp);
    // Root the unique dir at the CURRENT temp so any outer override (e.g. a
    // test harness's per-test scratch dir) is still honored.
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    std::filesystem::path unique;
    for (;;)
    {
      unique = base / ("neml2_aoti_" + std::to_string(counter.fetch_add(1)));
      std::error_code ec;
      if (std::filesystem::create_directory(unique, ec))
        break; // fresh, ours
      // Name already taken (stale dir, or a concurrent host process on the same
      // base): try the next counter value.
    }
    const std::string s = unique.string();
    _putenv_s("TMP", s.c_str());
    _putenv_s("TEMP", s.c_str());
  }

  ~ScopedExtractTempDir()
  {
    _restore("TMP", _old_tmp, _had_tmp);
    _restore("TEMP", _old_temp, _had_temp);
  }

  ScopedExtractTempDir(const ScopedExtractTempDir &) = delete;
  ScopedExtractTempDir & operator=(const ScopedExtractTempDir &) = delete;

private:
  static void _save(const char * name, std::string & out, bool & had)
  {
    const char * v = std::getenv(name); // copy immediately: _putenv may invalidate it
    had = (v != nullptr);
    if (had)
      out = v;
  }
  static void _restore(const char * name, const std::string & old, bool had)
  {
    // On Windows `_putenv_s(name, "")` deletes the variable; restore the prior
    // value, or delete it if it was unset when we started.
    _putenv_s(name, had ? old.c_str() : "");
  }

  std::string _old_tmp;
  std::string _old_temp;
  bool _had_tmp = false;
  bool _had_temp = false;
};

std::mutex &
extract_temp_mutex()
{
  static std::mutex m;
  return m;
}
#endif // _WIN32

// Build a loader with the consistent constructor args used across all AOTI
// artifacts produced by neml2-compile. `device_index` is -1 for cpu artifacts
// and cuda artifacts that target the current device; a concrete index pins a
// cuda artifact onto a specific GPU (used by the multi-device dispatcher). On
// Windows each load is given its own extraction temp dir (see
// ScopedExtractTempDir); on POSIX the construction is unchanged.
std::unique_ptr<torch::inductor::AOTIModelPackageLoader>
make_loader(const std::filesystem::path & path, int device_index = -1)
{
#ifdef _WIN32
  const std::lock_guard<std::mutex> lock(extract_temp_mutex());
  const ScopedExtractTempDir isolate_extract;
#endif
  return std::make_unique<torch::inductor::AOTIModelPackageLoader>(path.string(),
                                                                   /*model_name=*/"model",
                                                                   /*run_single_threaded=*/false,
                                                                   /*num_runners=*/1,
                                                                   device_index);
}

// Parse a `krylov` metadata block into a KrylovConfig (shared by the top-level
// forward-solve config and the per-segment sensitivity-solve descriptors).
void
parse_krylov_json(const nlohmann::json & kc, KrylovConfig & out)
{
  _assert(parse_krylov_method(kc.value("method", std::string("gmres")), out.method),
          "aoti::Model: unknown krylov method in metadata");
  _assert(parse_cache_strategy(kc.value("cache_strategy", std::string("none")), out.cache),
          "aoti::Model: unknown krylov cache_strategy in metadata");
  out.restart = kc.value("restart", out.restart);
  out.max_its = kc.value("max_its", out.max_its);
  out.abs_tol = kc.value("abs_tol", out.abs_tol);
  out.rel_tol = kc.value("rel_tol", out.rel_tol);
  out.cache_max_its = kc.value("cache_max_its", out.cache_max_its);
}

at::ScalarType
parse_dtype(const std::string & s)
{
  if (s == "float64")
    return at::kDouble;
  if (s == "float32")
    return at::kFloat;
  // Non-floating promoted parameters (integer / bool buffers) record their dtype
  // explicitly; floating parameters omit it and inherit the leaf dtype.
  if (s == "int64")
    return at::kLong;
  if (s == "int32")
    return at::kInt;
  if (s == "bool")
    return at::kBool;
  _assert(false, "aoti::Model: unsupported dtype '", s, "' in metadata.");
  return at::kDouble;
}

// The folder-name form (inverse of parse_dtype) used for the
// per-`<device>/<dtype>/` artifact leaf.
std::string
device_type_str(at::Device d)
{
  return d.is_cuda() ? "cuda" : "cpu";
}

std::string
dtype_str(at::ScalarType t)
{
  switch (t)
  {
    case at::kDouble:
      return "float64";
    case at::kFloat:
      return "float32";
    case at::kLong:
      return "int64";
    case at::kInt:
      return "int32";
    case at::kBool:
      return "bool";
    default:
      _assert(false, "aoti::Model: unsupported dtype for the artifact leaf path.");
      return "float64";
  }
}

// Boundary-rename helpers (used only when the artifact carries `boundary_aliases`).
// Re-key a name->tensor map through a translation map; a key absent from `tr`
// passes through unchanged (identity). Small maps, called once per public op.
std::map<std::string, at::Tensor>
rekey(const std::map<std::string, at::Tensor> & in, const std::map<std::string, std::string> & tr)
{
  std::map<std::string, at::Tensor> out;
  for (const auto & [k, v] : in)
  {
    auto it = tr.find(k);
    out.emplace(it == tr.end() ? k : it->second, v);
  }
  return out;
}

// Re-key a nested output->(input->tensor) Jacobian through an outer + inner
// translation (either may be empty => identity on that axis).
VariablePairJacobian
rekey_nested(const VariablePairJacobian & in,
             const std::map<std::string, std::string> & outer_tr,
             const std::map<std::string, std::string> & inner_tr)
{
  VariablePairJacobian out;
  for (const auto & [o, inner] : in)
  {
    auto oit = outer_tr.find(o);
    out.emplace(oit == outer_tr.end() ? o : oit->second, rekey(inner, inner_tr));
  }
  return out;
}
} // namespace

Model::Impl::Impl(const std::filesystem::path & artifact_root,
                  at::Device device,
                  at::ScalarType dtype)
{
  // Register any custom ops a compiled artifact may reference (e.g.
  // neml2::opaque_pow) before a segment loads/evaluates. Lazy + idempotent so it
  // does not collide with the Python package's registration (cpp-eager / py-aoti).
  ensure_neml2_custom_ops_registered();

  // Device + dtype come from the caller (the folder path), NOT the metadata --
  // one shared metadata.json backs every compiled (device, dtype) leaf.
  _device = device;
  _dtype = dtype;

  const auto meta_path = artifact_root / "metadata.json";
  std::ifstream meta_f(meta_path);
  _assert(static_cast<bool>(meta_f),
          "aoti::Model: failed to open metadata file '",
          meta_path.string(),
          "'. Path must point at the artifact root produced by `neml2-compile` "
          "(the folder containing metadata.json + <device>/<dtype>/ binaries).");
  const auto meta = nlohmann::json::parse(meta_f);

  // Wire-format handshake: the loader refuses any metadata whose schema_version
  // differs, so a stale cache fails with a clear "regenerate" message instead of
  // a cryptic missing-field error deep in the parser. The canonical value lives
  // in scripts/dependencies.yaml; this literal is kept in sync by dep_manager.py.
  // dependencies: aoti.schema_version
  static constexpr int kSupportedSchemaVersion = 13;
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

  // Concrete loader index: -1 lets the loader use the current device (cpu, or
  // the ambient cuda device); a concrete cuda index pins this Model to a GPU.
  const int dev_idx =
      (_device.is_cuda() && _device.has_index()) ? static_cast<int>(_device.index()) : -1;

  // .pt2 basenames are resolved against the per-(device, dtype) leaf. The leaf
  // must exist for the requested run; fail with the available leaves listed rather
  // than a cryptic missing-file error deep in the segment loader.
  const auto cache_dir = artifact_root / device_type_str(_device) / dtype_str(_dtype);
  if (!std::filesystem::is_directory(cache_dir))
  {
    std::string available;
    if (std::filesystem::is_directory(artifact_root))
      for (const auto & dev : std::filesystem::directory_iterator(artifact_root))
        if (dev.is_directory())
          for (const auto & dt : std::filesystem::directory_iterator(dev.path()))
            if (dt.is_directory())
            {
              if (!available.empty())
                available += ", ";
              available += dev.path().filename().string() + "/" + dt.path().filename().string();
            }
    _assert(false,
            "aoti::Model: no compiled artifact for device '",
            device_type_str(_device),
            "' + dtype '",
            dtype_str(_dtype),
            "' under '",
            artifact_root.string(),
            "' (looked in '",
            cache_dir.string(),
            "'). Available: ",
            available.empty() ? std::string("(none)") : available,
            ". Recompile with `neml2-compile --device ",
            device_type_str(_device),
            " --dtype ",
            dtype_str(_dtype),
            "`.");
  }

  // Boundary renames (shallow, optional). Only the names reported at the public
  // surface change; every internal structure keeps the ORIGINAL authored names.
  // Load the per-namespace forward / reverse maps now so the parameter loop can
  // store `_named_parameters` boundary-keyed and the ext views can be built
  // below. Absent => `_has_aliases` stays false and the runtime behaves exactly
  // as an unrenamed artifact.
  if (meta.contains("boundary_aliases"))
  {
    const auto & ba = meta["boundary_aliases"];
    auto load_pair_maps = [&ba](const char * key,
                                std::map<std::string, std::string> & o2e,
                                std::map<std::string, std::string> & e2o)
    {
      if (!ba.contains(key))
        return;
      for (auto it = ba[key].begin(); it != ba[key].end(); ++it)
      {
        const auto ext = it.value().get<std::string>();
        o2e[it.key()] = ext;
        e2o[ext] = it.key();
      }
    };
    load_pair_maps("inputs", _in_orig2ext, _in_ext2orig);
    load_pair_maps("outputs", _out_orig2ext, _out_ext2orig);
    if (ba.contains("parameters"))
      for (auto it = ba["parameters"].begin(); it != ba["parameters"].end(); ++it)
        _param_orig2ext[it.key()] = it.value().get<std::string>();
  }
  _has_aliases = !(_in_orig2ext.empty() && _out_orig2ext.empty() && _param_orig2ext.empty());

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

  // Pre-build the boundary (renamed) name vectors the public accessors return
  // (identity for any variable without a rename). Only when some rename is
  // active -- the unrenamed common case leaves these empty and the accessors
  // return the original vectors.
  if (_has_aliases)
  {
    _ext_input_names.reserve(_input_names.size());
    for (const auto & n : _input_names)
    {
      auto it = _in_orig2ext.find(n);
      _ext_input_names.push_back(it == _in_orig2ext.end() ? n : it->second);
    }
    _ext_output_names.reserve(_output_names.size());
    for (const auto & n : _output_names)
    {
      auto it = _out_orig2ext.find(n);
      _ext_output_names.push_back(it == _out_orig2ext.end() ? n : it->second);
    }
  }

  // Master (out, in) derivative pairs the artifact supports. Absent /
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

  // Master (out, param) parameter-derivative pairs the artifact supports.
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

  // Promoted parameters (empty in the common fully-baked case).
  if (meta.contains("parameters"))
  {
    for (const auto & p : meta["parameters"])
    {
      const auto name = p["name"].get<std::string>();
      // A floating parameter records NO dtype and inherits the leaf's (`_dtype`);
      // a non-floating parameter (int/bool buffer) records its own so it is not
      // coerced to the float type. Device is always the leaf device (`_device`,
      // which carries the concrete cuda index) -- no per-param device.
      const auto dtype = p.contains("dtype") ? parse_dtype(p["dtype"].get<std::string>()) : _dtype;
      const auto shape = p["shape"].get<std::vector<int64_t>>();
      // Natural base shape. Fallback to the full stored shape (== base for an
      // unbatched parameter) keeps older single-param paths robust.
      _param_base_shapes[name] = p.value("param_base_shape", shape);
      const auto opts = at::TensorOptions().dtype(dtype).device(_device);

      // Inline values vs. spilled state-dict reference.
      at::Tensor t;
      if (p.contains("values"))
      {
        const auto values = p["values"].get<std::vector<double>>();
        // Build on CPU first, then move to the leaf device/dtype. CUDA-side
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
      // `_named_parameters` is keyed by BOUNDARY name (identity when unaliased):
      // it is the mutable public surface AND the map the dispatcher slot-assigns
      // into for multi-device sync, and `_resolve_param` reads it back through
      // the same boundary key, so all mutation paths stay coherent. The natural
      // base shape stays keyed by the ORIGINAL name (every internal reader uses
      // original names); the boundary-keyed view is built below.
      _named_parameters.emplace(_param_boundary_name(name), t.contiguous());
    }
    // Boundary-keyed view of the natural base shapes for the public accessor
    // (`parameter_base_shapes()`), so it lines up with `named_parameters()`.
    if (_has_aliases)
      for (const auto & [orig, base] : _param_base_shapes)
        _ext_param_base_shapes[_param_boundary_name(orig)] = base;
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
    // `p` is the ORIGINAL param name (from `_param_derivatives`); look it up in
    // the boundary-keyed `_named_parameters` via its boundary name.
    auto pit = _named_parameters.find(_param_boundary_name(p));
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

  // Solver config: the implicit Newton's convergence / line-search settings ride
  // in the shared metadata so the Python-free runtime is configured straight from
  // the artifact. Absent (forward-only) => the SolverConfig defaults stand.
  // Verbosity is deliberately never recorded here: it is a diagnostic controlled
  // by the `NEML2_LOGS` env var / the log store (see log.h), independent of the
  // solver config. `Model::set_solver_config` still overrides these at runtime.
  if (meta.contains("solver_config"))
  {
    const auto & sc = meta["solver_config"];
    _solver_config.atol = sc.value("atol", _solver_config.atol);
    _solver_config.rtol = sc.value("rtol", _solver_config.rtol);
    _solver_config.miters = sc.value("miters", _solver_config.miters);
    _solver_config.ls_type = sc.value("ls_type", _solver_config.ls_type);
    _solver_config.ls_max_iters = sc.value("ls_max_iters", _solver_config.ls_max_iters);
    _solver_config.ls_cutback = sc.value("ls_cutback", _solver_config.ls_cutback);
    _solver_config.ls_c = sc.value("ls_c", _solver_config.ls_c);
    _solver_config.substep_del_tol = sc.value("substep_del_tol", _solver_config.substep_del_tol);

    // Linear-solver kind (schema v11). "direct" (default) chains jacobian ->
    // solve; "krylov" runs a matrix-free Krylov solve over the matvec graph,
    // configured by the nested `krylov` block.
    _solver_kind = sc.value("solver_kind", std::string("direct"));
    if (sc.contains("krylov"))
      parse_krylov_json(sc["krylov"], _krylov_config);
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
      // Substep role/pair, present on implicit-segment givens.
      if (v.contains("role") && !v["role"].is_null())
        info.role = v["role"].get<std::string>();
      if (v.contains("pair") && !v["pair"].is_null())
        info.pair = v["pair"].get<std::string>();
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
      // Per-(out_var, in_var) Jacobian-pair metadata. Present iff the segment
      // also packaged a JVP loader; orders one entry per trailing pair tensor
      // in the JVP loader's output tuple (row-major: outputs outer, structural
      // inputs inner).
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
          pi.row_offset = p.value("row_offset", static_cast<int64_t>(0));
          pi.col_offset = p.value("col_offset", static_cast<int64_t>(0));
          seg.jacobian_pairs.push_back(std::move(pi));
        }
      }
      // Parameter-derivative graphs (present iff parameter derivatives were
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
          pi.row_offset = p.value("row_offset", static_cast<int64_t>(0));
          pi.col_offset = p.value("col_offset", static_cast<int64_t>(0));
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
      seg.max_substepping_level = seg_meta.value("max_substepping_level", 0);
      // Operator + solve graphs (schema v10+): the Jacobian is emitted as an
      // operator and the linear solve is a separate graph the C++ driver chains.
      // Which graphs are present depends on the solver kind -- a direct solve has
      // jacobian + solve; a Krylov solve has matvec (+ jacobian only when a
      // preconditioner / input derivative needs A). Load each iff its key exists.
      seg.residual_loader =
          make_loader(cache_dir / seg_meta["residual_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("jacobian_package"))
        seg.jacobian_loader =
            make_loader(cache_dir / seg_meta["jacobian_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("solve_package"))
        seg.solve_loader =
            make_loader(cache_dir / seg_meta["solve_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("matvec_package"))
        seg.matvec_loader =
            make_loader(cache_dir / seg_meta["matvec_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("precond_setup_package"))
      {
        seg.precond_setup_loader =
            make_loader(cache_dir / seg_meta["precond_setup_package"].get<std::string>(), dev_idx);
        seg.precond_apply_loader =
            make_loader(cache_dir / seg_meta["precond_apply_package"].get<std::string>(), dev_idx);
      }
      // IFT: `jacobian_given` (B = ∂r/∂g) is emitted whenever an input derivative
      // is compiled; `solve_ift` (the baked direct solve) only when the
      // input-sensitivity solver is direct -- an iterative sensitivity solve
      // (`input_sensitivity_kind == "krylov"`) runs `krylov_solve_dense` over the
      // assembled A instead (schema v12).
      if (seg_meta.contains("jacobian_given_package"))
        seg.jacobian_given_loader =
            make_loader(cache_dir / seg_meta["jacobian_given_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("solve_ift_package"))
        seg.solve_ift_loader =
            make_loader(cache_dir / seg_meta["solve_ift_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("input_sensitivity"))
      {
        const auto & isd = seg_meta["input_sensitivity"];
        seg.input_sensitivity_kind = isd.value("kind", std::string("direct"));
        if (isd.contains("krylov"))
          parse_krylov_json(isd["krylov"], seg.input_sensitivity_krylov);
      }

      for (const auto & v : seg_meta["unknowns"])
        seg.unknowns.push_back(parse_var_info(v));
      for (const auto & v : seg_meta["givens"])
        seg.givens.push_back(parse_var_info(v));
      for (const auto & v : seg_meta["residuals"])
        seg.residuals.push_back(parse_var_info(v));

      // Per-group metadata for the per-group I/O contract. Newton inner loop
      // runs entirely per-group; per-var ↔ per-group conv happens twice per
      // solve via _pack_groups / _unpack_groups.
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

      // Per-(unknown, given) IFT Jacobian-pair metadata. The IFT loader emits
      // one block per pair (via AssembledMatrix.disassemble), in jacobian_pairs
      // order, so the runtime composes them against dg_dmaster exactly like a
      // forward segment's per-pair blocks.
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
          pi.row_offset = p.value("row_offset", static_cast<int64_t>(0));
          pi.col_offset = p.value("col_offset", static_cast<int64_t>(0));
          seg.jacobian_pairs.push_back(std::move(pi));
        }
      }
      // Implicit parameter-derivative graph (ParamIFT). Present iff a parameter
      // derivative targeting a promoted parameter inside the residual was
      // requested. Emits one dense du/dθ block per (unknown, param) pair in
      // param_jacobian_pairs order (unknowns outer, params inner).
      // ParamIFT: `dr_dparam` (dense A + ∂r/∂θ) is emitted whenever a parameter
      // derivative is compiled; `solve_param` (the baked direct solve) only when
      // the param-sensitivity solver is direct -- an iterative sensitivity solve
      // (`param_sensitivity_kind == "krylov"`) runs `krylov_solve_dense` over the
      // dense A instead (schema v12).
      if (seg_meta.contains("dr_dparam_package"))
        seg.dr_dparam_loader =
            make_loader(cache_dir / seg_meta["dr_dparam_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("solve_param_package"))
        seg.solve_param_loader =
            make_loader(cache_dir / seg_meta["solve_param_package"].get<std::string>(), dev_idx);
      if (seg_meta.contains("param_jacobian_pairs"))
      {
        for (const auto & p : seg_meta["param_jacobian_pairs"])
        {
          Segment::ParamPairInfo pi;
          pi.out_var = p["out_var"].get<std::string>();
          pi.param = p["param"].get<std::string>();
          if (p.contains("out_base_shape"))
            pi.out_base_shape = p["out_base_shape"].get<std::vector<int64_t>>();
          if (p.contains("param_base_shape"))
            pi.param_base_shape = p["param_base_shape"].get<std::vector<int64_t>>();
          pi.row_offset = p.value("row_offset", static_cast<int64_t>(0));
          pi.col_offset = p.value("col_offset", static_cast<int64_t>(0));
          seg.param_jacobian_pairs.push_back(std::move(pi));
        }
      }
      if (seg_meta.contains("param_sensitivity"))
      {
        const auto & psd = seg_meta["param_sensitivity"];
        seg.param_sensitivity_kind = psd.value("kind", std::string("direct"));
        if (psd.contains("krylov"))
          parse_krylov_json(psd["krylov"], seg.param_sensitivity_krylov);
      }
      // Solver convergence / line-search configuration is read once from the
      // top-level `solver_config` in metadata.json into `_solver_config` below;
      // `Model::set_solver_config` still overrides it at runtime.
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
  // `name` is the ORIGINAL segment/metadata name; `_named_parameters` and the
  // per-call `_param_overrides` are keyed by BOUNDARY name (identity when the
  // parameter is unaliased), so map it through first.
  const std::string & key = _param_boundary_name(name);
  if (_param_overrides != nullptr)
  {
    auto oit = _param_overrides->find(key);
    if (oit != _param_overrides->end())
      return oit->second;
  }
  auto it = _named_parameters.find(key);
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

Model::Model(const std::filesystem::path & artifact_root, at::Device device, at::ScalarType dtype)
  : _impl(std::make_unique<Impl>(artifact_root, device, dtype))
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

// The ops run through `_guarded` so every exception leaving the public surface
// is a neml2 Exception with a meaningful `recoverable()`: a recoverable
// ConvergenceError from the Newton solve passes through, while a foreign torch
// error (e.g. a shape / device mismatch raised inside a compiled graph) is
// normalized to a non-recoverable FatalError.
//
// When the artifact carries boundary renames (`_has_aliases`) each op re-keys at
// the interface: incoming input / tangent dicts BOUNDARY->original, cotangents
// BOUNDARY->original (keyed by output name), and results original->BOUNDARY.
// `param_overrides` passes through unchanged -- it is keyed by boundary name and
// `_resolve_param` maps each original segment name to its boundary key. The
// unrenamed common case takes the no-copy fast path.
std::map<std::string, at::Tensor>
Model::forward(const std::map<std::string, at::Tensor> & inputs,
               const std::map<std::string, at::Tensor> & param_overrides) const
{
  return _guarded(
      [&]() -> std::map<std::string, at::Tensor>
      {
        if (!_impl->_has_aliases)
          return _impl->forward(inputs, param_overrides);
        auto out = _impl->forward(rekey(inputs, _impl->_in_ext2orig), param_overrides);
        return rekey(out, _impl->_out_orig2ext);
      });
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::jvp(const std::map<std::string, at::Tensor> & inputs,
           const std::map<std::string, at::Tensor> & tangents,
           const std::map<std::string, at::Tensor> & param_overrides) const
{
  using Ret = std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>;
  return _guarded(
      [&]() -> Ret
      {
        if (!_impl->_has_aliases)
          return _impl->jvp(inputs, tangents, param_overrides);
        auto [out, jout] = _impl->jvp(rekey(inputs, _impl->_in_ext2orig),
                                      rekey(tangents, _impl->_in_ext2orig),
                                      param_overrides);
        return {rekey(out, _impl->_out_orig2ext), rekey(jout, _impl->_out_orig2ext)};
      });
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::jacobian(const std::map<std::string, at::Tensor> & inputs,
                const std::map<std::string, at::Tensor> & param_overrides) const
{
  using Ret = std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>;
  return _guarded(
      [&]() -> Ret
      {
        if (!_impl->_has_aliases)
          return _impl->jacobian(inputs, param_overrides);
        auto [out, jac] = _impl->jacobian(rekey(inputs, _impl->_in_ext2orig), param_overrides);
        return {rekey(out, _impl->_out_orig2ext),
                rekey_nested(jac, _impl->_out_orig2ext, _impl->_in_orig2ext)};
      });
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::param_jacobian(const std::map<std::string, at::Tensor> & inputs,
                      const std::map<std::string, at::Tensor> & param_overrides) const
{
  using Ret = std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>;
  return _guarded(
      [&]() -> Ret
      {
        if (!_impl->_has_aliases)
          return _impl->param_jacobian(inputs, param_overrides);
        auto [out, pjac] =
            _impl->param_jacobian(rekey(inputs, _impl->_in_ext2orig), param_overrides);
        // Inner keys are promoted-parameter names -> boundary via _param_orig2ext.
        return {rekey(out, _impl->_out_orig2ext),
                rekey_nested(pjac, _impl->_out_orig2ext, _impl->_param_orig2ext)};
      });
}

std::map<std::string, at::Tensor>
Model::param_vjp(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & cotangents,
                 const std::map<std::string, at::Tensor> & param_overrides) const
{
  return _guarded(
      [&]() -> std::map<std::string, at::Tensor>
      {
        if (!_impl->_has_aliases)
          return _impl->param_vjp(inputs, cotangents, param_overrides);
        // cotangents are keyed by OUTPUT name (BOUNDARY->original); the result is
        // keyed by promoted-parameter name (original->BOUNDARY).
        auto grads = _impl->param_vjp(rekey(inputs, _impl->_in_ext2orig),
                                      rekey(cotangents, _impl->_out_ext2orig),
                                      param_overrides);
        return rekey(grads, _impl->_param_orig2ext);
      });
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
