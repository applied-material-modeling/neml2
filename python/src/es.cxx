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

#include <pybind11/pybind11.h>

#include "neml2/neml2.h"

#include "csrc/es/types.h"

namespace py = pybind11;
using namespace neml2;

PYBIND11_MODULE(es, m)
{
  m.doc() = "Wrappers and utilities for NEML2 equation systems";

  // Force PyTorch Python-side initialization early
  py::module_::import("torch");
  // Ensure tensor types are registered for Tensor field conversions
  py::module_::import("neml2.tensors");

  // declare py classes (all declared before any def() call so that
  // forward-referenced types are already registered)
  auto cls_AxisLayout =
      py::class_<AxisLayout>(m, "AxisLayout", "Layout of variables along an equation-system axis");
  auto cls_SparseVector = py::class_<SparseVector>(
      m, "SparseVector", "Sparse representation of a vector as a list of tensors and their layout");
  auto cls_SparseMatrix =
      py::class_<SparseMatrix>(m,
                               "SparseMatrix",
                               "Sparse representation of a matrix as a 2D list of tensors and "
                               "their layout");
  auto cls_AssembledVector = py::class_<AssembledVector>(
      m, "AssembledVector", "Dense vector assembled from a list of tensors and their layout");
  auto cls_AssembledMatrix =
      py::class_<AssembledMatrix>(m,
                                  "AssembledMatrix",
                                  "Dense matrix assembled from a 2D list of tensors and their "
                                  "layout");
  auto cls_NonlinearSystem =
      py::class_<ModelNonlinearSystem, std::shared_ptr<ModelNonlinearSystem>>(
          m, "NonlinearSystem", "Nonlinear system wrapper for models.");

  // free functions
  m.def(
      "load_nonlinear_system",
      [](const py::object & path, const std::string & name)
      {
        auto factory = load_input(py::str(path).cast<std::string>());
        return factory->get_es<ModelNonlinearSystem>(name);
      },
      py::arg("path"),
      py::arg("name"),
      R"(
  A convenient function to load an input file and get a nonlinear system.

  This function is equivalent to calling core.load_input followed by
  Factory.get_nonlinear_system. Note that this convenient function does not
  support passing additional command-line arguments and will force the creation of a new
  es.NonlinearSystem even if one has already been created. Use core.load_input and
  Factory.get_nonlinear_system if you need finer control over the model creation behavior.

  :param path:      Path to the input file to be parsed
  :param name:      Name of the nonlinear system
  )");

  // binding definitions
  def(m, cls_AxisLayout);
  def(m, cls_SparseVector);
  def(m, cls_SparseMatrix);
  def(m, cls_AssembledVector);
  def(m, cls_AssembledMatrix);
  def(m, cls_NonlinearSystem);
}
