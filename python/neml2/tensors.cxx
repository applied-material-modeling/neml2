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

#include "neml2/tensors/tensors.h"

#include <pybind11/pybind11.h>

#include "python/neml2/csrc/tensors/types.h"

namespace py = pybind11;
using namespace neml2;

PYBIND11_MODULE(tensors, m)
{
  m.doc() = "NEML2 primitive tensor types";

  // Export the Number type
  m.attr("Number") = py::module_::import("torch.types").attr("Number");

  // Export enums
  auto tensor_type_enum = py::enum_<TensorType>(m, "TensorType");
#define TENSORTYPE_ENUM_ENTRY(T) tensor_type_enum.value(#T, TensorType::k##T)
  FOR_ALL_TENSORBASE(TENSORTYPE_ENUM_ENTRY);

// Declare all the TensorBase derived tensors
// This is as simple as calling py::class_, but it is important to do this for ALL tensors up
// front. The reason is for typing: Once we build the neml2 python library with all the necessary
// bindings, we will have to extract all the typing information (mostly function signature) from
// the library, which is needed by language servers like Pylance. We use pybind11-stubgen for that
// purpose. For a type to be deducible by pybind11-stubgen, a concrete definition of the binding
// class must exist at the point of method definition. Therefore, we need to first create all the
// class definitions before creating method bindings that use them as arguments.
#define DECL_PYTYPE(T) auto cls_##T = py::class_<T>(m, #T)
  FOR_ALL_TENSORBASE(DECL_PYTYPE);

// Next, actually define the bindings for each tensor type
#define DEF_PYTYPE(T) def(m, cls_##T)
  FOR_ALL_TENSORBASE(DEF_PYTYPE);

  // Define operator bindings
  def_operators(m);
}
