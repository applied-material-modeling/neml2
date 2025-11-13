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

#include "python/neml2/tensors/BaseView.h"
#include "python/neml2/tensors/types.h"

namespace py = pybind11;
using namespace neml2;

template <class T>
void
def_BaseView(py::module_ & m, const std::string & name)
{
  auto c = py::class_<BaseView<T>>(m, name.c_str());
  c.def(py::init<T *>())
      .def("dim", &BaseView<T>::dim)
      .def_property_readonly("shape",
                             [](const BaseView<T> & self)
                             {
                               const auto s = self.sizes();
                               py::tuple pys(s.size());
                               for (std::size_t i = 0; i < s.size(); i++)
                                 pys[i] = s[i];
                               return pys;
                             })
      .def("size", &BaseView<T>::size)
      .def("__getitem__", &BaseView<T>::index)
      .def("__getitem__",
           [](BaseView<T> * self, at::indexing::TensorIndex index)
           { return self->index({std::move(index)}); })
      .def("slice", &BaseView<T>::slice)
      .def("__setitem__", &BaseView<T>::index_put_)
      .def("__setitem__",
           [](BaseView<T> * self, at::indexing::TensorIndex index, const ATensor & src)
           { return self->index_put_({std::move(index)}, src); })
      .def("__setitem__",
           [](BaseView<T> * self, const indexing::TensorIndices & indices, const Tensor & src)
           { return self->index_put_(indices, src); })
      .def("__setitem__",
           [](BaseView<T> * self, at::indexing::TensorIndex index, const Tensor & src)
           { return self->index_put_({std::move(index)}, src); })
      .def("expand", py::overload_cast<TensorShapeRef>(&BaseView<T>::expand, py::const_))
      .def("expand", py::overload_cast<Size, Size>(&BaseView<T>::expand, py::const_))
      .def("expand_as", &BaseView<T>::expand_as)
      .def("reshape", &BaseView<T>::reshape)
      .def("squeeze", &BaseView<T>::squeeze)
      .def("unsqueeze", &BaseView<T>::unsqueeze, py::arg("dim"), py::arg("n") = 1)
      .def("transpose", &BaseView<T>::transpose)
      .def("movedim", &BaseView<T>::movedim)
      .def("flatten", &BaseView<T>::flatten);
}

template <class T>
BaseView<T>::BaseView(T * data)
  : _data(data)
{
}

template <class T>
Size
BaseView<T>::dim() const
{
  return _data->base_dim();
}

template <class T>
TensorShapeRef
BaseView<T>::sizes() const
{
  return _data->base_sizes();
}

template <class T>
Size
BaseView<T>::size(Size i) const
{
  return _data->base_size(i);
}

template <class T>
Tensor
BaseView<T>::index(const indexing::TensorIndices & indices) const
{
  return _data->base_index(indices);
}

template <class T>
Tensor
BaseView<T>::slice(Size dim, const indexing::Slice & slice) const
{
  return _data->base_slice(dim, slice);
}

template <class T>
void
BaseView<T>::index_put_(const indexing::TensorIndices & indices, const ATensor & src)
{
  _data->base_index_put_(indices, src);
}

template <class T>
Tensor
BaseView<T>::expand(TensorShapeRef new_shape) const
{
  return _data->base_expand(new_shape);
}

template <class T>
Tensor
BaseView<T>::expand(Size dim, Size new_size) const
{
  return _data->base_expand(dim, new_size);
}

template <class T>
Tensor
BaseView<T>::expand_as(const Tensor & other) const
{
  return _data->base_expand_as(other);
}

template <class T>
Tensor
BaseView<T>::reshape(TensorShapeRef new_shape) const
{
  return _data->base_reshape(new_shape);
}

template <class T>
Tensor
BaseView<T>::squeeze(Size dim) const
{
  return _data->base_squeeze(dim);
}

template <class T>
Tensor
BaseView<T>::unsqueeze(Size dim, Size n) const
{
  return _data->base_unsqueeze(dim, n);
}

template <class T>
Tensor
BaseView<T>::transpose(Size d1, Size d2) const
{
  return _data->base_transpose(d1, d2);
}

template <class T>
Tensor
BaseView<T>::movedim(Size source, Size destination) const
{
  return _data->base_movedim(source, destination);
}

template <class T>
Tensor
BaseView<T>::flatten() const
{
  return _data->base_flatten();
}

#define INSTANTIATE_BASEVIEW(T)                                                                    \
  template void def_BaseView<T>(py::module_ & m, const std::string & name);                        \
  template class BaseView<T>
FOR_ALL_TENSORBASE(INSTANTIATE_BASEVIEW);
