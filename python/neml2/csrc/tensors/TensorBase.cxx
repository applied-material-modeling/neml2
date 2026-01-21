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

#include <memory>

#include "python/neml2/csrc/tensors/TensorBase.h"
#include "python/neml2/csrc/tensors/DynamicView.h"
#include "python/neml2/csrc/tensors/IntmdView.h"
#include "python/neml2/csrc/tensors/BaseView.h"
#include "python/neml2/csrc/tensors/BatchView.h"
#include "python/neml2/csrc/tensors/StaticView.h"

namespace py = pybind11;
using namespace neml2;

template <class T>
void
def_TensorBase(py::module_ & m, const std::string & type)
{
  def_DynamicView<T>(m, (type + "DynamicView").c_str());
  def_IntmdView<T>(m, (type + "IntmdView").c_str());
  def_BaseView<T>(m, (type + "BaseView").c_str());
  def_BatchView<T>(m, (type + "BatchView").c_str());
  def_StaticView<T>(m, (type + "StaticView").c_str());

  auto c = m.attr(type.c_str()).cast<py::class_<T>>();
  c.def(py::init<>())
      .def(py::init<const ATensor &, Size, Size>(),
           py::arg("tensor"),
           py::arg("dynamic_dim"),
           py::arg("intmd_dim"))
      .def("__str__",
           [type](const T & self)
           {
             return utils::stringify(self) + '\n' + "<" + type + " of shape " +
                    utils::stringify(self.dynamic_sizes()) + utils::stringify(self.intmd_sizes()) +
                    utils::stringify(self.base_sizes()) + ">";
           })
      .def("__repr__",
           [type](const T & self)
           {
             return "<" + type + " of shape " + utils::stringify(self.dynamic_sizes()) +
                    utils::stringify(self.intmd_sizes()) + utils::stringify(self.base_sizes()) +
                    ">";
           })
      .def("torch", [](const T & self) { return torch::Tensor(self); })
      .def("tensor", [](const T & self) { return neml2::Tensor(self); })
      .def_property_readonly("dynamic",
                             [](py::object self)
                             {
                               auto view = py::cast(DynamicView<T>(self.cast<T *>()));
                               py::detail::keep_alive_impl(view, self);
                               return view;
                             })
      .def_property_readonly("intmd",
                             [](py::object self)
                             {
                               auto view = py::cast(IntmdView<T>(self.cast<T *>()));
                               py::detail::keep_alive_impl(view, self);
                               return view;
                             })
      .def_property_readonly("base",
                             [](py::object self)
                             {
                               auto view = py::cast(BaseView<T>(self.cast<T *>()));
                               py::detail::keep_alive_impl(view, self);
                               return view;
                             })
      .def_property_readonly("batch",
                             [](py::object self)
                             {
                               auto view = py::cast(BatchView<T>(self.cast<T *>()));
                               py::detail::keep_alive_impl(view, self);
                               return view;
                             })
      .def_property_readonly("static",
                             [](py::object self)
                             {
                               auto view = py::cast(StaticView<T>(self.cast<T *>()));
                               py::detail::keep_alive_impl(view, self);
                               return view;
                             })
      .def("contiguous", &T::contiguous)
      .def("clone", &T::clone)
      .def("detach", &T::detach)
      .def("detach_", &T::detach_)
      .def(
          "to",
          [](T * self, NEML2_TENSOR_OPTIONS_VARGS) { return self->to(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def("copy_", &T::copy_)
      .def("zero_", &T::zero_)
      .def_property_readonly("requires_grad", &T::requires_grad)
      .def("requires_grad_", &T::requires_grad_)
      .def("defined", &T::defined)
      .def_property_readonly("dtype", &T::scalar_type)
      .def_property_readonly("device", &T::device)
      .def("dim", &T::dim)
      .def_property_readonly("shape", &T::sizes)
      .def_property_readonly("grad", &T::grad)
      .def("item", [](const T & self) { return self.item(); });

  // convert from another tensor
#define DEF_COPY_CONSTRUCTOR(T2) c.def(py::init<const T2 &>(), py::arg("other"))
  FOR_ALL_TENSORBASE(DEF_COPY_CONSTRUCTOR);

  // Static methods
  c.def_static("empty_like", &T::empty_like)
      .def_static("zeros_like", &T::zeros_like)
      .def_static("ones_like", &T::ones_like)
      .def_static(
          "full_like",
          [](const T & other, double init) { return T::full_like(other, init); },
          py::arg("other"),
          py::arg("fill_value"))
      .def_static("rand_like", &T::rand_like);
}

// Explicit template instantiations
#define INSTANTIATE_TENSORBASE(T)                                                                  \
  template void def_TensorBase<T>(py::module_ &, const std::string &)
FOR_ALL_TENSORBASE(INSTANTIATE_TENSORBASE);
