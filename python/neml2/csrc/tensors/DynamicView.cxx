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

#include "python/neml2/csrc/tensors/DynamicView.h"
#include "python/neml2/csrc/tensors/types.h"

namespace py = pybind11;
using namespace neml2;

template <class T>
void
def_DynamicView(py::module_ & m, const std::string & name)
{
  auto c = py::class_<DynamicView<T>>(m, name.c_str());
  c.def(py::init<T *>())
      .def("dim", &DynamicView<T>::dim)
      .def_property_readonly("shape",
                             [](const DynamicView<T> & self)
                             {
                               const auto s = self.sizes();
                               py::tuple pys(s.size());
                               for (std::size_t i = 0; i < s.size(); i++)
                                 pys[i] = s[i];
                               return pys;
                             })
      .def("size", &DynamicView<T>::size)
      .def("__getitem__", &DynamicView<T>::index)
      .def("__getitem__",
           [](DynamicView<T> * self, at::indexing::TensorIndex index)
           { return self->index({std::move(index)}); })
      .def("slice", &DynamicView<T>::slice)
      .def("__setitem__", &DynamicView<T>::index_put_)
      .def("__setitem__",
           [](DynamicView<T> * self, at::indexing::TensorIndex index, const ATensor & src)
           { return self->index_put_({std::move(index)}, src); })
      .def("__setitem__",
           [](DynamicView<T> * self, const indexing::TensorIndices & indices, const Tensor & src)
           { return self->index_put_(indices, src); })
      .def("__setitem__",
           [](DynamicView<T> * self, at::indexing::TensorIndex index, const Tensor & src)
           { return self->index_put_({std::move(index)}, src); })
      .def("expand", py::overload_cast<TensorShapeRef>(&DynamicView<T>::expand, py::const_))
      .def("expand", py::overload_cast<Size, Size>(&DynamicView<T>::expand, py::const_))
      .def("expand_as", &DynamicView<T>::expand_as)
      .def("reshape", &DynamicView<T>::reshape)
      .def("squeeze", &DynamicView<T>::squeeze)
      .def("unsqueeze", &DynamicView<T>::unsqueeze, py::arg("dim"), py::arg("n") = 1)
      .def("transpose", &DynamicView<T>::transpose)
      .def("movedim", &DynamicView<T>::movedim)
      .def("flatten", &DynamicView<T>::flatten);
}

template <class T>
DynamicView<T>::DynamicView(T * data)
  : _data(data)
{
}

template <class T>
Size
DynamicView<T>::dim() const
{
  return _data->dynamic_dim();
}

template <class T>
TensorShape
DynamicView<T>::sizes() const
{
  return _data->dynamic_sizes().concrete();
}

template <class T>
Size
DynamicView<T>::size(Size i) const
{
  return _data->dynamic_size(i).concrete();
}

template <class T>
T
DynamicView<T>::index(const indexing::TensorIndices & indices) const
{
  return _data->dynamic_index(indices);
}

template <class T>
T
DynamicView<T>::slice(Size dim, const indexing::Slice & slice) const
{
  return _data->dynamic_slice(dim, slice);
}

template <class T>
void
DynamicView<T>::index_put_(const indexing::TensorIndices & indices, const ATensor & src)
{
  _data->dynamic_index_put_(indices, src);
}

template <class T>
T
DynamicView<T>::expand(TensorShapeRef new_shape) const
{
  return _data->dynamic_expand(new_shape);
}

template <class T>
T
DynamicView<T>::expand(Size dim, Size new_size) const
{
  return _data->dynamic_expand(dim, new_size);
}

template <class T>
T
DynamicView<T>::expand_as(const Tensor & other) const
{
  return _data->dynamic_expand_as(other);
}

template <class T>
T
DynamicView<T>::reshape(TensorShapeRef new_shape) const
{
  return _data->dynamic_reshape(new_shape);
}

template <class T>
T
DynamicView<T>::squeeze(Size dim) const
{
  return _data->dynamic_squeeze(dim);
}

template <class T>
T
DynamicView<T>::unsqueeze(Size dim, Size n) const
{
  return _data->dynamic_unsqueeze(dim, n);
}

template <class T>
T
DynamicView<T>::transpose(Size d1, Size d2) const
{
  return _data->dynamic_transpose(d1, d2);
}

template <class T>
T
DynamicView<T>::movedim(Size source, Size destination) const
{
  return _data->dynamic_movedim(source, destination);
}

template <class T>
T
DynamicView<T>::flatten() const
{
  return _data->dynamic_flatten();
}

#define INSTANTIATE_DYNAMICVIEW(T)                                                                 \
  template void def_DynamicView<T>(py::module_ & m, const std::string & name);                     \
  template class DynamicView<T>
FOR_ALL_TENSORBASE(INSTANTIATE_DYNAMICVIEW);
