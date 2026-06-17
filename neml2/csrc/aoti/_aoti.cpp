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
// AOTI-exported NEML2 model artifacts (.pt2 + _meta.json). Lives under
// python/neml2/native/aoti/ so it's adjacent to its Python-side neighbors
// (loader helpers, drivers) and stays isolated from the legacy bindings in
// python/src/, which will eventually be retired.

#include <filesystem>
#include <map>
#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "neml2/csrc/aoti/Model.h"

namespace py = pybind11;
using neml2::aoti::Model;

PYBIND11_MODULE(_aoti, m)
{
  m.doc() = "Pybind11 binding for neml2::aoti::Model. The bare C++ runtime "
            "loads AOTI-exported NEML2 model artifacts (.pt2 + _meta.json) "
            "and exposes three operations -- forward / jvp / jacobian -- "
            "plus a mutable named_parameters() surface for the entries that "
            "were promoted at compile time via `neml2-compile --parameter`.";

  // Force PyTorch's pybind tensor caster initialization.
  py::module_::import("torch");

  py::class_<Model>(m, "Model", R"(
Thin C++ runtime for an AOTI-exported NEML2 model.

Construct from the path to the metadata JSON produced by ``neml2-compile``;
the loader resolves the per-segment ``.pt2`` files relative to that path.

The artifact is device- and dtype-pinned at export time; there is no
runtime ``to()``. To target a different device, re-run ``neml2-compile``.

Parameters that were explicitly promoted via ``--parameter NAME`` at
compile time are reachable through ``named_parameters()`` and may be
mutated in-place (e.g. ``model.named_parameters()['E'].fill_(210000.0)``).
Everything else is baked into the graph as a constant.
)")
      // Take ``meta_path`` as ``std::string`` (rather than
      // ``std::filesystem::path`` via the stl/filesystem caster) so the
      // pybind11-stubgen-generated annotation comes out as ``str``
      // instead of ``os.PathLike``. The current stubgen release
      // (≤2.5.5) emits ``os.PathLike`` without an accompanying
      // ``import os``, which trips pyright; pybind/pybind11-stubgen#280
      // fixes this upstream, drop this lambda + restore
      // ``py::init<const std::filesystem::path &>()`` once a release
      // with that PR lands.
      .def(py::init([](const std::string & meta_path)
                    { return std::make_unique<Model>(std::filesystem::path{meta_path}); }),
           py::arg("meta_path"),
           "Load all .pt2 segments + metadata from `meta_path`. Throws on "
           "any missing file or schema mismatch.")
      .def_property_readonly(
          "input_names", &Model::input_names, "Master input names in graph-call order.")
      .def_property_readonly(
          "output_names", &Model::output_names, "Master output names in graph-call order.")
      .def_property_readonly("input_sizes",
                             &Model::input_sizes,
                             "Per-input flat sizes (product of declared base shape; 1 for Scalar).")
      .def_property_readonly(
          "output_sizes",
          &Model::output_sizes,
          "Per-output flat sizes (product of declared base shape; 1 for Scalar).")
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
          [](const Model & m, const std::map<std::string, at::Tensor> & inputs)
          {
            // ``Model::forward`` returns ``std::map`` which is sorted by key;
            // re-pack into a Python dict in ``output_names`` declaration
            // order so the caller can rely on ``list(outs.keys()) ==
            // model.output_names()`` for tuple-style consumers.
            auto out_map = m.forward(inputs);
            py::dict result;
            for (const auto & name : m.output_names())
              result[name.c_str()] = out_map.at(name);
            return result;
          },
          py::arg("inputs"),
          R"(
Evaluate the model.

``inputs`` is keyed by the names returned by ``input_names``; missing
keys raise an error. Returns one tensor per name in ``output_names``,
preserving declaration order.
)")
      .def("jvp",
           &Model::jvp,
           py::arg("inputs"),
           py::arg("tangents"),
           R"(
Evaluate + JVP.

``tangents`` shares its keys with ``inputs``; missing keys default to
zero. Returns a 2-tuple ``(outputs, jvp_outputs)`` -- both ``dict[str,
Tensor]`` keyed by ``output_names``.
)")
      .def("jacobian",
           &Model::jacobian,
           py::arg("inputs"),
           R"(
Evaluate + full Jacobian.

Returns a 2-tuple ``(outputs, J)`` where ``J`` is the assembled
``(*B, sum(output_sizes), sum(input_sizes))`` block-stacked Jacobian
over the **structural** inputs (promoted-parameter inputs are not
exposed in J).
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
          {
            auto & params = self.named_parameters();
            auto it = params.find(name);
            if (it == params.end())
              throw py::key_error("set_parameter: '" + name +
                                  "' is not a promoted parameter; only entries that appear in "
                                  "named_parameters() may be set.");
            it->second = value.contiguous();
          },
          py::arg("name"),
          py::arg("value"),
          "Replace a promoted parameter's tensor (the C++-side slot is updated).");
}
