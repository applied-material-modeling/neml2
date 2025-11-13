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

#include "python/neml2/tensors/BatchView.h"
#include "python/neml2/tensors/types.h"

namespace py = pybind11;
using namespace neml2;

template <class T>
void
def_BatchView(py::module_ & m, const std::string & name)
{
  auto c = py::class_<BatchView<T>>(m, name.c_str());
  c.def(py::init<T *>())
      .def("dim", &BatchView<T>::dim)
      .def_property_readonly("shape",
                             [](const BatchView<T> & self)
                             {
                               const auto s = self.sizes();
                               py::tuple pys(s.size());
                               for (std::size_t i = 0; i < s.size(); i++)
                                 pys[i] = s[i];
                               return pys;
                             })
      .def("size", &BatchView<T>::size)
      .def("expand", &BatchView<T>::expand)
      .def("expand_as", &BatchView<T>::expand_as)
      .def("reshape", &BatchView<T>::reshape)
      .def("flatten", &BatchView<T>::flatten);
}

template <class T>
BatchView<T>::BatchView(T * data)
  : _data(data)
{
}

template <class T>
Size
BatchView<T>::dim() const
{
  return _data->batch_dim();
}

template <class T>
TensorShape
BatchView<T>::sizes() const
{
  return _data->batch_sizes().concrete();
}

template <class T>
Size
BatchView<T>::size(Size i) const
{
  return _data->batch_size(i).concrete();
}

template <class T>
T
BatchView<T>::expand(TensorShapeRef new_dynamic_shape, TensorShapeRef new_intmd_shape) const
{
  return _data->batch_expand(new_dynamic_shape, new_intmd_shape);
}

template <class T>
T
BatchView<T>::expand_as(const Tensor & other) const
{
  return _data->batch_expand_as(other);
}

template <class T>
T
BatchView<T>::reshape(TensorShapeRef new_dynamic_shape, TensorShapeRef new_intmd_shape) const
{
  return _data->batch_reshape(new_dynamic_shape, new_intmd_shape);
}

template <class T>
T
BatchView<T>::flatten() const
{
  return _data->batch_flatten();
}

#define INSTANTIATE_BATCHVIEW(T)                                                                   \
  template void def_BatchView<T>(py::module_ & m, const std::string & name);                       \
  template class BatchView<T>
FOR_ALL_TENSORBASE(INSTANTIATE_BATCHVIEW);
