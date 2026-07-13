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

// Pybind11 binding for neml2::aoti::Model -- the thin C++ runtime that loads
// AOTI-exported NEML2 model artifacts (shared metadata.json + per-device/.pt2).
// Lives under python/neml2/native/aoti/ so it's adjacent to its Python-side
// neighbors (loader helpers, drivers) and stays isolated from the legacy
// bindings in python/src/, which will eventually be retired.

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/aoti/assertions.h"
#include "neml2/csrc/aoti/newton.h"
#include "neml2/csrc/aoti/nonlinear_system_eager.h"

namespace py = pybind11;
using neml2::aoti::Model;

PYBIND11_MODULE(_aoti, m)
{
  m.doc() = "Pybind11 binding for neml2::aoti::Model. The bare C++ runtime "
            "loads AOTI-exported NEML2 model artifacts (shared metadata.json "
            "+ per-<device>/<dtype>/ .pt2 binaries) and exposes three "
            "operations -- forward / jvp / jacobian -- plus a mutable "
            "named_parameters() surface for the entries that were promoted at "
            "compile time via `neml2-compile --parameter`.";

  // Force PyTorch's pybind tensor caster initialization.
  py::module_::import("torch");

  // Surface the recoverable ConvergenceError as a dedicated Python type so the
  // recoverable/fatal split survives the C++ -> Python translation. Subclasses
  // RuntimeError, so existing ``except RuntimeError`` handlers keep working;
  // code that needs the distinction (e.g. the eager runtime re-raising a
  // recoverable neml2::aoti::ConvergenceError after the C++ -> Python -> C++
  // round trip; see neml2/csrc/eager/Model.cpp) can match it precisely. Without
  // this, pybind's default translator collapses every neml2::aoti::Exception
  // (a std::runtime_error) to a bare RuntimeError, losing recoverable().
  py::register_exception<neml2::aoti::ConvergenceError>(m, "ConvergenceError", PyExc_RuntimeError);

  py::class_<Model>(m, "Model", R"(
Thin C++ runtime for an AOTI-exported NEML2 model.

Construct from the artifact root directory produced by ``neml2-compile``:
one shared ``metadata.json`` plus per-``<device>/<dtype>/`` ``.pt2`` binaries.
The loader selects the leaf matching the requested device and dtype.

The artifact is device- and dtype-pinned at export time; there is no
runtime ``to()``. To target a different device, re-run ``neml2-compile``.

Parameters that were explicitly promoted via ``--parameter NAME`` at
compile time are reachable through ``named_parameters()`` and may be
mutated in-place (e.g. ``model.named_parameters()['E'].fill_(210000.0)``).
Everything else is baked into the graph as a constant.
)")
      // Take ``artifact_root`` as ``std::string`` (rather than
      // ``std::filesystem::path`` via the stl/filesystem caster) so the
      // pybind11-stubgen-generated annotation comes out as ``str``
      // instead of ``os.PathLike``. The current stubgen release
      // (≤2.5.5) emits ``os.PathLike`` without an accompanying
      // ``import os``, which trips pyright; pybind/pybind11-stubgen#280
      // fixes this upstream, drop this lambda + restore
      // ``py::init<const std::filesystem::path &, ...>()`` once a
      // release with that PR lands.
      .def(py::init(
               [](const std::string & artifact_root,
                  std::optional<std::string> device,
                  std::optional<std::string> dtype)
               {
                 // device/dtype default to NEML2's canonical cpu/float64 leaf --
                 // matching `neml2-compile` and the py-aoti shim -- NOT torch's
                 // ambient defaults: an AOTI artifact is device/dtype-pinned
                 // (float64-first), and torch's ambient dtype is float32, which
                 // would never match a stock artifact. Explicit strings override
                 // (e.g. "float32", or a concrete "cuda:1").
                 std::string dev = device.value_or("cpu");
                 std::string dt = dtype.value_or("float64");
                 if (dev.empty())
                   dev = "cpu";
                 if (dt.empty())
                   dt = "float64";
                 // The leaf dtype is always a float type (the model dtype); a small
                 // local map keeps the binding free of the anon-namespace parser in
                 // Model.cpp. device uses torch's own string ctor (cpu/cuda/cuda:1).
                 at::ScalarType st;
                 if (dt == "float64")
                   st = at::kDouble;
                 else if (dt == "float32")
                   st = at::kFloat;
                 else
                   throw std::runtime_error("AOTIModel: unsupported dtype '" + dt +
                                            "' (expected float64 or float32).");
                 return std::make_unique<Model>(
                     std::filesystem::path{artifact_root}, at::Device{dev}, st);
               }),
           py::arg("artifact_root"),
           py::arg("device") = py::none(),
           py::arg("dtype") = py::none(),
           "Load the compiled artifact rooted at `artifact_root` (shared "
           "metadata.json + <device>/<dtype>/ binaries). `device`/`dtype` default "
           "to NEML2's canonical cpu/float64 (matching neml2-compile), not torch's "
           "ambient defaults. Throws on any missing file or schema mismatch.")
      .def_property_readonly(
          "input_names", &Model::input_names, "Master input names in graph-call order.")
      .def_property_readonly(
          "output_names", &Model::output_names, "Master output names in graph-call order.")
      .def_property_readonly(
          "input_base_shapes",
          &Model::input_base_shapes,
          "Per-input base shape (Scalar -> [], SR2 -> [6], R2 -> [3, 3]). Inputs "
          "must be passed at their canonical (*B, *base_shape) shape.")
      .def_property_readonly("output_base_shapes",
                             &Model::output_base_shapes,
                             "Per-output base shape (Scalar -> [], SR2 -> [6], R2 -> [3, 3]).")
      .def_property_readonly(
          "parameter_base_shapes",
          &Model::parameter_base_shapes,
          "Per-promoted-parameter natural base shape, keyed by qualified name "
          "(Scalar -> [], SR2 -> [6]). The keys are the promoted parameters (same "
          "keys as named_parameters()); empty when nothing was promoted. The "
          "parameter analogue of input_base_shapes / output_base_shapes, and the "
          "unified parameter surface shared with neml2.eager._EagerModel.")
      .def_property_readonly(
          "device",
          [](const Model & m) { return m.device(); },
          "Device the artifact was compiled for (immutable).")
      .def_property_readonly(
          "dtype",
          [](const Model & m) { return m.dtype(); },
          "Floating-point dtype the artifact was compiled for (immutable).")
      .def(
          "forward",
          [](const Model & m,
             const std::map<std::string, at::Tensor> & inputs,
             const std::map<std::string, at::Tensor> & param_overrides)
          {
            // ``Model::forward`` returns ``std::map`` which is sorted by key;
            // re-pack into a Python dict in ``output_names`` declaration
            // order so the caller can rely on ``list(outs.keys()) ==
            // model.output_names()`` for tuple-style consumers.
            auto out_map = m.forward(inputs, param_overrides);
            py::dict result;
            for (const auto & name : m.output_names())
              result[name.c_str()] = out_map.at(name);
            return result;
          },
          py::arg("inputs"),
          py::arg("param_overrides") = std::map<std::string, at::Tensor>{},
          R"(
Evaluate the model.

``inputs`` is keyed by the names returned by ``input_names``; missing
keys raise an error. Returns one tensor per name in ``output_names``,
preserving declaration order. ``param_overrides`` (default empty) replaces a
promoted parameter's value for this call only, without mutating
``named_parameters()`` -- a hook for multi-device dispatch.
)")
      .def("jvp",
           &Model::jvp,
           py::arg("inputs"),
           py::arg("tangents"),
           py::arg("param_overrides") = std::map<std::string, at::Tensor>{},
           R"(
Evaluate + JVP.

``inputs`` and ``tangents`` are keyed by ``input_names`` and shaped
``(*B, *base_shape)``; a missing tangent key defaults to zero. Returns a
2-tuple ``(outputs, jvp_outputs)`` -- both ``dict[str, Tensor]`` keyed by
``output_names``; ``jvp_outputs[name]`` is the directional derivative at the
output's natural ``(*B, *out_base_shape)``.
)")
      .def("jacobian",
           &Model::jacobian,
           py::arg("inputs"),
           py::arg("param_overrides") = std::map<std::string, at::Tensor>{},
           R"(
Evaluate + full Jacobian as unflattened variable-pair blocks.

Returns a 2-tuple ``(outputs, J)`` where ``J`` is a nested
``dict[str, dict[str, Tensor]]``: ``J[out_name][in_name]`` is the block
``(*B, *out_base_shape, *in_base_shape)`` (e.g. SR2->SR2 -> (*B, 6, 6);
Scalar->SR2 -> (*B, 6)) over the **structural** inputs (promoted-parameter
inputs are not exposed in J).
)")
      .def("param_jacobian",
           &Model::param_jacobian,
           py::arg("inputs"),
           py::arg("param_overrides") = std::map<std::string, at::Tensor>{},
           R"(
Evaluate + parameter Jacobian as unflattened variable-pair blocks.

Returns a 2-tuple ``(outputs, P)`` where ``P`` is a nested
``dict[str, dict[str, Tensor]]``: ``P[out_name][param_qname]`` is the block
``(*B, *out_base_shape, *param_base_shape)`` (e.g. stress w.r.t. a Scalar E ->
(*B, 6)). The keys are promoted parameters (see ``named_parameters()``), not
structural inputs. Requires the artifact was compiled with ``-d OUT:PARAM`` over
a promoted parameter; otherwise raises.
)")
      .def("param_vjp",
           &Model::param_vjp,
           py::arg("inputs"),
           py::arg("cotangents"),
           py::arg("param_overrides") = std::map<std::string, at::Tensor>{},
           R"(
Parameter VJP / adjoint: ``dL/d(param)`` for the loss
``L = sum_o <cotangent_o, out_o>``.

``cotangents`` is a ``dict[str, Tensor]`` keyed by output name, each at the
output's ``(*B, *out_base_shape)`` shape. Returns ``dict[str, Tensor]`` keyed by
parameter qualified name -- the cheaper form for many-parameter inverse
optimization. Same compile requirement as ``param_jacobian``.
)")
      .def(
          "named_parameters",
          [](Model & self) -> std::map<std::string, at::Tensor> &
          { return self.named_parameters(); },
          py::return_value_policy::reference_internal,
          R"(
Return the mutable map of runtime-flexible (promoted) parameters.

The dict's tensor values share storage with the C++-side parameter slots;
in-place mutation (``model.named_parameters()['E'].fill_(...)``) is
reflected on the next ``forward`` / ``jvp`` / ``jacobian`` call. Reassigning
a dict entry via ``model.named_parameters()['E'] = new_tensor`` updates
the *Python* dict only, not the C++ slot -- use ``set_parameter`` for that.

Empty when the model was compiled with no ``--parameter`` flags.
)")
      .def(
          "set_parameter",
          [](Model & self, const std::string & name, const at::Tensor & value)
          { self.set_parameter(name, value); },
          py::arg("name"),
          py::arg("value"),
          "Replace a promoted parameter's tensor (the C++-side slot is updated).")
      .def(
          "set_solver_config",
          [](Model & self,
             double atol,
             double rtol,
             std::size_t miters,
             const std::string & ls_type,
             std::size_t ls_max_iters,
             double ls_cutback,
             double ls_c)
          {
            neml2::aoti::SolverConfig cfg;
            cfg.atol = atol;
            cfg.rtol = rtol;
            cfg.miters = miters;
            cfg.ls_type = ls_type;
            cfg.ls_max_iters = ls_max_iters;
            cfg.ls_cutback = ls_cutback;
            cfg.ls_c = ls_c;
            self.set_solver_config(cfg);
          },
          py::arg("atol"),
          py::arg("rtol"),
          py::arg("miters"),
          py::arg("ls_type"),
          py::arg("ls_max_iters"),
          py::arg("ls_cutback"),
          py::arg("ls_c"),
          "Configure the implicit-segment Newton solve (override the values "
          "read from metadata.json at load time).");

  // Eager-path entry point: the same C++ Newton solver the AOTI runtime uses,
  // driven over Python-supplied residual/step callables (RHS / (Jacobian -> LinearSolve)).
  // This is what unifies the eager solve with the compiled one -- a single
  // iteration-control implementation.
  m.def(
      "newton_solve_eager",
      [](py::object residual_fn,
         py::object step_fn,
         std::vector<std::pair<std::string, std::vector<int64_t>>> unknown_layout,
         std::vector<std::pair<std::string, std::vector<int64_t>>> residual_layout,
         std::vector<at::Tensor> u0,
         double atol,
         double rtol,
         std::size_t miters,
         const std::string & ls_type,
         std::size_t ls_max_iters,
         double ls_cutback,
         double ls_c,
         bool collect_log)
      {
        neml2::aoti::SolverConfig cfg;
        cfg.atol = atol;
        cfg.rtol = rtol;
        cfg.miters = miters;
        cfg.ls_type = ls_type;
        cfg.ls_max_iters = ls_max_iters;
        cfg.ls_cutback = ls_cutback;
        cfg.ls_c = ls_c;
        cfg.collect_log = collect_log;
        return neml2::aoti::run_eager_newton(cfg,
                                             std::move(residual_fn),
                                             std::move(step_fn),
                                             std::move(unknown_layout),
                                             std::move(residual_layout),
                                             std::move(u0));
      },
      py::arg("residual_fn"),
      py::arg("step_fn"),
      py::arg("unknown_layout"),
      py::arg("residual_layout"),
      py::arg("u0"),
      py::arg("atol"),
      py::arg("rtol"),
      py::arg("miters"),
      py::arg("ls_type"),
      py::arg("ls_max_iters"),
      py::arg("ls_cutback"),
      py::arg("ls_c"),
      py::arg("collect_log") = false,
      R"(
Run the shared C++ Newton solver over an eager (Python-delegating) system.

``residual_fn(list[Tensor]) -> list[Tensor]`` and ``step_fn(list[Tensor]) ->
(list[Tensor], list[Tensor])`` supply the per-group residual and Newton step
(they bind the givens + linear solver, e.g. ``RHS`` + ``Jacobian`` -> ``LinearSolve``).
``unknown_layout`` / ``residual_layout`` are ``(structure, sub_batch_shape)``
per group. Returns ``(u_solved, converged, iterations)``.
)");

  // Eager iterative path: the SAME C++ Newton loop, but each Newton step's linear
  // solve is a matrix-free Krylov iteration (krylov.h) instead of a direct solve.
  // The outer Newton config drives convergence/line-search; the krylov_* args
  // configure the inner solve (method / restart / preconditioner / cache).
  m.def(
      "krylov_solve_eager",
      [](py::object residual_fn,
         py::object matvec_fn,
         py::object jacobian_fn,
         std::vector<std::pair<std::string, std::vector<int64_t>>> unknown_layout,
         std::vector<std::pair<std::string, std::vector<int64_t>>> residual_layout,
         std::vector<int64_t> block_sizes,
         std::vector<at::Tensor> u0,
         double atol,
         double rtol,
         std::size_t miters,
         const std::string & ls_type,
         std::size_t ls_max_iters,
         double ls_cutback,
         double ls_c,
         bool collect_log,
         const std::string & method,
         int64_t restart,
         int64_t max_its,
         double abs_tol,
         double rel_tol,
         const std::string & preconditioner,
         const std::string & cache_strategy,
         int64_t cache_max_its)
      {
        neml2::aoti::SolverConfig ncfg;
        ncfg.atol = atol;
        ncfg.rtol = rtol;
        ncfg.miters = miters;
        ncfg.ls_type = ls_type;
        ncfg.ls_max_iters = ls_max_iters;
        ncfg.ls_cutback = ls_cutback;
        ncfg.ls_c = ls_c;
        ncfg.collect_log = collect_log;

        neml2::aoti::KrylovConfig kcfg;
        neml2::aoti::_assert(neml2::aoti::parse_krylov_method(method, kcfg.method),
                             "krylov_solve_eager: unknown method '",
                             method,
                             "' (expected gmres | bicgstab)");
        neml2::aoti::_assert(neml2::aoti::parse_precond_kind(preconditioner, kcfg.precond),
                             "krylov_solve_eager: unknown preconditioner '",
                             preconditioner,
                             "' (expected none | jacobi | block_jacobi | full)");
        neml2::aoti::_assert(neml2::aoti::parse_cache_strategy(cache_strategy, kcfg.cache),
                             "krylov_solve_eager: unknown cache_strategy '",
                             cache_strategy,
                             "' (expected none | chord | max_its)");
        kcfg.restart = restart;
        kcfg.max_its = max_its;
        kcfg.abs_tol = abs_tol;
        kcfg.rel_tol = rel_tol;
        kcfg.cache_max_its = cache_max_its;

        return neml2::aoti::run_eager_krylov(ncfg,
                                             kcfg,
                                             std::move(residual_fn),
                                             std::move(matvec_fn),
                                             std::move(jacobian_fn),
                                             std::move(unknown_layout),
                                             std::move(residual_layout),
                                             std::move(block_sizes),
                                             std::move(u0));
      },
      py::arg("residual_fn"),
      py::arg("matvec_fn"),
      py::arg("jacobian_fn"),
      py::arg("unknown_layout"),
      py::arg("residual_layout"),
      py::arg("block_sizes"),
      py::arg("u0"),
      py::arg("atol"),
      py::arg("rtol"),
      py::arg("miters"),
      py::arg("ls_type"),
      py::arg("ls_max_iters"),
      py::arg("ls_cutback"),
      py::arg("ls_c"),
      py::arg("collect_log") = false,
      py::arg("method") = "gmres",
      py::arg("restart") = 40,
      py::arg("max_its") = 1000,
      py::arg("abs_tol") = 0.0,
      py::arg("rel_tol") = 1.0e-4,
      py::arg("preconditioner") = "none",
      py::arg("cache_strategy") = "none",
      py::arg("cache_max_its") = 10,
      R"(
Run the shared C++ Newton solver over an eager system whose inner linear solve is
a matrix-free Krylov iteration (GMRES / BiCGStab) rather than a direct solve.

``residual_fn(u) -> b`` and ``matvec_fn(u, v) -> Jv`` supply the residual and the
matrix-free ``J.v`` (the eager ``RHS`` / ``Matvec`` modules). ``jacobian_fn(u) ->
Tensor`` returns the dense operator ``(*B, N, N)`` for the preconditioner (pass a
callable only when ``preconditioner != "none"``). ``block_sizes`` are the
per-variable widths of the single dense unknown group (BlockJacobi). Returns
``(u_solved, converged, iterations, log)``.
)");
}
