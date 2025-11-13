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

#include "python/neml2/tensors/StaticView.h"
#include "python/neml2/core/types.h"
#include "python/neml2/core/utils.h"

namespace py = pybind11;
using namespace neml2;

template <class T>
void
def_StaticView(py::module_ & m, const std::string & name)
{
  // import types
  get_pycls<Tensor>(m, "Tensor");

  auto c = py::class_<StaticView<T>>(m, name.c_str());
  c.def(py::init<T *>())
      .def("dim", &StaticView<T>::dim)
      .def_property_readonly("shape",
                             [](const StaticView<T> & self)
                             {
                               const auto s = self.sizes();
                               py::tuple pys(s.size());
                               for (std::size_t i = 0; i < s.size(); i++)
                                 pys[i] = s[i];
                               return pys;
                             })
      .def("size", &StaticView<T>::size)
      .def("expand", &StaticView<T>::expand)
      .def("expand_as", &StaticView<T>::expand_as)
      .def("reshape", &StaticView<T>::reshape)
      .def("flatten", &StaticView<T>::flatten);
}

template <class T>
StaticView<T>::StaticView(T * data)
  : _data(data)
{
}

template <class T>
Size
StaticView<T>::dim() const
{
  return _data->static_dim();
}

template <class T>
TensorShapeRef
StaticView<T>::sizes() const
{
  return _data->static_sizes();
}

template <class T>
Size
StaticView<T>::size(Size i) const
{
  return _data->static_size(i);
}

template <class T>
Tensor
StaticView<T>::expand(TensorShapeRef new_intmd_shape, TensorShapeRef new_base_shape) const
{
  return _data->static_expand(new_intmd_shape, new_base_shape);
}

template <class T>
Tensor
StaticView<T>::expand_as(const Tensor & other) const
{
  return _data->static_expand_as(other);
}

template <class T>
Tensor
StaticView<T>::reshape(TensorShapeRef new_intmd_shape, TensorShapeRef new_base_shape) const
{
  return _data->static_reshape(new_intmd_shape, new_base_shape);
}

template <class T>
Tensor
StaticView<T>::flatten() const
{
  return _data->static_flatten();
}

#define INSTANTIATE_STATICVIEW(T)                                                                  \
  template void def_StaticView<T>(py::module_ & m, const std::string & name);                      \
  template class StaticView<T>
FOR_ALL_TENSORBASE(INSTANTIATE_STATICVIEW);
