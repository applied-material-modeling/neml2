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

#include "python/neml2/tensors/DynamicView.h"
#include "python/neml2/core/types.h"

namespace py = pybind11;
using namespace neml2;

template <class Derived>
void
def_DynamicView(py::module_ & m, const std::string & name)
{
  // "forward" declarations
  py::object py_tensor_cls = m.attr("Tensor");
  py::class_<Tensor> tensor_cls(py_tensor_cls);

  auto c = py::class_<DynamicView<Derived>>(m, name.c_str());
  c.def(py::init<Derived *>())
      .def("dim", &DynamicView<Derived>::dim)
      .def_property_readonly("shape",
                             [](const DynamicView<Derived> & self)
                             {
                               const auto s = self.sizes();
                               py::tuple pys(s.size());
                               for (std::size_t i = 0; i < s.size(); i++)
                                 pys[i] = s[i];
                               return pys;
                             })
      .def("size", &DynamicView<Derived>::size)
      .def("__getitem__", &DynamicView<Derived>::index)
      .def("__getitem__",
           [](DynamicView<Derived> * self, at::indexing::TensorIndex index)
           { return self->index({std::move(index)}); })
      .def("slice", &DynamicView<Derived>::slice)
      .def("__setitem__", &DynamicView<Derived>::index_put_)
      .def("__setitem__",
           [](DynamicView<Derived> * self, at::indexing::TensorIndex index, const ATensor & src)
           { return self->index_put_({std::move(index)}, src); })
      .def("__setitem__",
           [](DynamicView<Derived> * self,
              const indexing::TensorIndices & indices,
              const Tensor & src) { return self->index_put_(indices, src); })
      .def("__setitem__",
           [](DynamicView<Derived> * self, at::indexing::TensorIndex index, const Tensor & src)
           { return self->index_put_({std::move(index)}, src); })
      .def("expand", py::overload_cast<TensorShapeRef>(&DynamicView<Derived>::expand, py::const_))
      .def("expand", py::overload_cast<Size, Size>(&DynamicView<Derived>::expand, py::const_))
      .def("expand_as", &DynamicView<Derived>::expand_as)
      .def("reshape", &DynamicView<Derived>::reshape)
      .def("squeeze", &DynamicView<Derived>::squeeze)
      .def("unsqueeze", &DynamicView<Derived>::unsqueeze, py::arg("dim"), py::arg("n") = 1)
      .def("transpose", &DynamicView<Derived>::transpose)
      .def("movedim", &DynamicView<Derived>::movedim)
      .def("flatten", &DynamicView<Derived>::flatten);
}

template <class Derived>
DynamicView<Derived>::DynamicView(Derived * data)
  : _data(data)
{
}

template <class Derived>
Size
DynamicView<Derived>::dim() const
{
  return _data->dynamic_dim();
}

template <class Derived>
TensorShape
DynamicView<Derived>::sizes() const
{
  return _data->dynamic_sizes().concrete();
}

template <class Derived>
Size
DynamicView<Derived>::size(Size i) const
{
  return _data->dynamic_size(i).concrete();
}

template <class Derived>
Derived
DynamicView<Derived>::index(const indexing::TensorIndices & indices) const
{
  return _data->dynamic_index(indices);
}

template <class Derived>
Derived
DynamicView<Derived>::slice(Size dim, const indexing::Slice & slice) const
{
  return _data->dynamic_slice(dim, slice);
}

template <class Derived>
void
DynamicView<Derived>::index_put_(const indexing::TensorIndices & indices, const ATensor & src)
{
  _data->dynamic_index_put_(indices, src);
}

template <class Derived>
Derived
DynamicView<Derived>::expand(TensorShapeRef new_shape) const
{
  return _data->dynamic_expand(new_shape);
}

template <class Derived>
Derived
DynamicView<Derived>::expand(Size dim, Size new_size) const
{
  return _data->dynamic_expand(dim, new_size);
}

template <class Derived>
Derived
DynamicView<Derived>::expand_as(const Tensor & other) const
{
  return _data->dynamic_expand_as(other);
}

template <class Derived>
Derived
DynamicView<Derived>::reshape(TensorShapeRef new_shape) const
{
  return _data->dynamic_reshape(new_shape);
}

template <class Derived>
Derived
DynamicView<Derived>::squeeze(Size dim) const
{
  return _data->dynamic_squeeze(dim);
}

template <class Derived>
Derived
DynamicView<Derived>::unsqueeze(Size dim, Size n) const
{
  return _data->dynamic_unsqueeze(dim, n);
}

template <class Derived>
Derived
DynamicView<Derived>::transpose(Size d1, Size d2) const
{
  return _data->dynamic_transpose(d1, d2);
}

template <class Derived>
Derived
DynamicView<Derived>::movedim(Size source, Size destination) const
{
  return _data->dynamic_movedim(source, destination);
}

template <class Derived>
Derived
DynamicView<Derived>::flatten() const
{
  return _data->dynamic_flatten();
}

#define INSTANTIATE_DYNAMICVIEW(T)                                                                 \
  template void def_DynamicView<T>(py::module_ & m, const std::string & name);                     \
  template class DynamicView<T>
FOR_ALL_TENSORBASE(INSTANTIATE_DYNAMICVIEW);
