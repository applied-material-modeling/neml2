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

#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/base/LabeledAxis.h"
#include "neml2/tensors/TensorValue.h"
#include "neml2/base/Factory.h"
#include "neml2/models/Model.h"
#include "neml2/models/Assembler.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

void def_LabeledAxisAccessor(pybind11::module_ &);
void def_LabeledAxis(pybind11::module_ &);
void def_TensorValue(pybind11::module_ &);
void def_Factory(pybind11::module_ &);
void def_Model(pybind11::module_ &);
void def_VectorAssembler(pybind11::module_ &);
void def_MatrixAssembler(pybind11::module_ &);

PYBIND11_MODULE(core, m)
{
  m.doc() = "NEML2 Python bindings";

  // declare bindings
  auto c1 = pybind11::class_<neml2::LabeledAxisAccessor>(m, "LabeledAxisAccessor");
  auto c2 = pybind11::class_<neml2::LabeledAxis>(m, "LabeledAxis");
  auto c3 = pybind11::class_<neml2::TensorValueBase>(
      m,
      "TensorValue",
      "The interface for working with tensor values (parameters, "
      "buffers, etc.) managed by models.");
  auto c4 = pybind11::class_<neml2::Factory>(
      m, "Factory", "Factory for creating objects defined in the input file");
  auto c5 = pybind11::class_<neml2::Model, std::shared_ptr<neml2::Model>>(
      m, "Model", "The canonical type for constitutive models in NEML2.");
  auto c6 = pybind11::class_<neml2::VectorAssembler>(m, "VectorAssembler");
  auto c7 = pybind11::class_<neml2::MatrixAssembler>(m, "MatrixAssembler");

  // free functions
  m.def("load_input",
        &neml2::load_input,
        pybind11::arg("path"),
        pybind11::arg("cli_args") = "",
        R"(
  Parse all options from an input file. Note that Previously loaded input options
  will be discarded.

  :param path:     Path to the input file to be parsed
  :param cli_args: Additional command-line arguments to pass to the parser
  )");
  m.def("load_model",
        &neml2::load_model,
        pybind11::arg("path"),
        pybind11::arg("name"),
        R"(
  A convenient function to load an input file and get a model.

  This function is equivalent to calling core.load_input followed by
  Factory.get_model. Note that this convenient function does not support passing
  additional command-line arguments and will force the creation of a new
  core.Model even if one has already been created. Use core.load_input and
  Factory.get_model if you need finer control over the model creation behavior.

  :param path:      Path to the input file to be parsed
  :param name:      Name of the model
  )");
  m.def(
      "diagnose",
      [](const neml2::Model & m)
      {
        auto diagnoses = diagnose(m);
        std::vector<std::string> issues;
        issues.reserve(diagnoses.size());
        for (const auto & diagnosis : diagnoses)
          issues.emplace_back(diagnosis.what());
        return issues;
      },
      pybind11::arg("model"),
      R"(
  Diagnose common issues in model setup. Raises a runtime error including all identified issues,
  if any.

  :param model: Model to be diagnosed
  )");

  // binding definitions
  def_LabeledAxisAccessor(m);
  def_LabeledAxis(m);
  def_TensorValue(m);
  def_Factory(m);
  def_Model(m);
  def_VectorAssembler(m);
  def_MatrixAssembler(m);
}
