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

#pragma once

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <torch/python.h>

#include "python/neml2/indexing.h"
#include "python/neml2/types.h"
#include "neml2/tensors/LabeledTensor.h"

namespace py = pybind11;

namespace neml2
{
// Forward declarations
template <class Derived>
void def_LabeledBatchView(py::module_ & m, const std::string & name);
template <class Derived>
void def_LabeledBaseView(py::module_ & m, const std::string & name);
template <class Derived, Size D>
void def_LabeledTensor(py::class_<Derived> & c);

/**
 * @brief Convenient shim wrapper for working with batch dimensions of LabeledTensor
 *
 * The view does NOT extend the life of of the wrapped tensor.
 */
template <class Derived>
class LabeledBatchView
{
public:
  LabeledBatchView(Derived * data)
    : _data(data)
  {
  }
  // These methods mirror LabeledTensor (the batch_xxx ones)
  Size dim() const { return _data->batch_dim(); }
  TensorShapeRef sizes() const { return _data->batch_sizes(); }
  Derived index(const indexing::TensorIndices & i) const { return _data->batch_index(i); }
  void index_put_(const indexing::TensorIndices & i, const torch::Tensor & t)
  {
    _data->batch_index_put_(i, t);
  }

private:
  Derived * _data;
};

/**
 * @brief Convenient shim wrapper for working with base dimensions of LabeledTensor
 *
 * The view does NOT extend the life of of the wrapped tensor.
 */
template <class Derived>
class LabeledBaseView
{
public:
  LabeledBaseView(Derived * data)
    : _data(data)
  {
  }
  // These methods mirror LabeledTensor (the batch_xxx ones)
  Size dim() const { return _data->base_dim(); }
  TensorShapeRef sizes() const { return _data->base_sizes(); }
  Tensor index(const indexing::TensorLabels & i) const { return _data->base_index(i); }
  void index_put_(const indexing::TensorLabels & i, const Tensor & t)
  {
    _data->base_index_put_(i, t);
  }
  Size storage() const { return _data->base_storage(); }

private:
  Derived * _data;
};

} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
template <class Derived>
void
def_LabeledBatchView(py::module_ & m, const std::string & name)
{
  auto c = py::class_<LabeledBatchView<Derived>>(m, name.c_str())
               .def(py::init<Derived *>())
               .def("dim", &LabeledBatchView<Derived>::dim)
               .def_property_readonly("shape", &LabeledBatchView<Derived>::sizes)
               .def("__getitem__", &LabeledBatchView<Derived>::index)
               .def("__getitem__",
                    [](LabeledBatchView<Derived> * self, at::indexing::TensorIndex index)
                    { return self->index({index}); })
               // Setters using torch::Tensor
               .def("__setitem__", &LabeledBatchView<Derived>::index_put_)
               .def("__setitem__",
                    [](LabeledBatchView<Derived> * self,
                       at::indexing::TensorIndex index,
                       const torch::Tensor & src) { return self->index_put_({index}, src); })
               // Setters using neml2::Tensor
               .def("__setitem__",
                    [](LabeledBatchView<Derived> * self,
                       const indexing::TensorIndices & indices,
                       const Tensor & src) { return self->index_put_(indices, src); })
               .def("__setitem__",
                    [](LabeledBatchView<Derived> * self,
                       at::indexing::TensorIndex index,
                       const Tensor & src) { return self->index_put_({index}, src); });
}

template <class Derived>
void
def_LabeledBaseView(py::module_ & m, const std::string & name)
{
  auto c =
      py::class_<LabeledBaseView<Derived>>(m, name.c_str())
          .def(py::init<Derived *>())
          .def("dim", &LabeledBaseView<Derived>::dim)
          .def_property_readonly("shape", &LabeledBaseView<Derived>::sizes)
          .def("__getitem__", &LabeledBaseView<Derived>::index)
          .def("__getitem__",
               [](LabeledBaseView<Derived> * self, indexing::TensorLabel index)
               { return self->index({index}); })
          // Setters using neml2::Tensor
          .def("__setitem__",
               [](LabeledBaseView<Derived> * self,
                  indexing::TensorLabels indices,
                  const Tensor & src) { self->index_put_(indices, src); })
          .def("__setitem__",
               [](LabeledBaseView<Derived> * self, indexing::TensorLabel index, const Tensor & src)
               { return self->index_put_({index}, src); })
          .def("storage", &LabeledBaseView<Derived>::storage);
}

template <class Derived, Size D>
void
def_LabeledTensor(py::class_<Derived> & c)
{
  // Ctors, conversions, accessors etc.
  c.def(py::init<>())
      .def(py::init<const torch::Tensor &, const std::array<const LabeledAxis *, D> &>())
      .def(py::init<const Tensor &, const std::array<const LabeledAxis *, D> &>())
      .def(py::init<const Derived &>())
      .def("__str__",
           [](const Derived & self)
           {
             std::ostringstream os;
             for (Size i = 0; i < D; i++)
               os << "Axis " << i << ":\n" << self.axis(i) << "\n\n";
             os << self.tensor() << '\n';
             os << "Batch shape: " << self.batch_sizes() << '\n';
             os << " Base shape: " << self.base_sizes() << '\n';
             return os.str();
           })
      .def("torch", [](const Derived & self) { return torch::Tensor(self); })
      .def("tensor", [](const Derived & self) { return Tensor(self); })
      .def_property_readonly("axes", &Derived::axes, py::return_value_policy::reference)
      .def_property_readonly("batch",
                             [](Derived * self) { return new LabeledBatchView<Derived>(self); })
      .def_property_readonly("base",
                             [](Derived * self) { return new LabeledBaseView<Derived>(self); })
      .def("clone", [](Derived * self) { return self->clone(); })
      .def("detach", &Derived::detach)
      .def("detach_", &Derived::detach_)
      .def(
          "to",
          [](Derived * self, NEML2_TENSOR_OPTIONS_VARGS) { return self->to(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def("copy_", &Derived::copy_)
      .def("zero_", &Derived::zero_)
      .def("dim", &Derived::dim)
      .def_property_readonly("shape", &Derived::sizes)
      .def("batched", &Derived::batched)
      .def_property_readonly("dtype", &Derived::scalar_type)
      .def_property_readonly("device", &Derived::device)
      .def("requires_grad_", &Derived::requires_grad_)
      .def_property_readonly("requires_grad", &Derived::requires_grad)
      .def_property_readonly("grad", [](Derived * self) { return self->tensor().grad(); })
      .def("fill",
           &Derived::fill,
           py::arg("other"),
           py::arg("odim") = 0,
           py::arg("recursive") = true);

  // Static methods
  c.def_static(
       "empty",
       [](const TensorShapeRef & batch_shape,
          const std::array<const LabeledAxis *, D> & axes,
          NEML2_TENSOR_OPTIONS_VARGS)
       { return Derived::empty(batch_shape, axes, NEML2_TENSOR_OPTIONS); },
       py::arg("batch_shape"),
       py::arg("axes"),
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](const TensorShapeRef & batch_shape,
             const std::array<const LabeledAxis *, D> & axes,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return Derived::zeros(batch_shape, axes, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("axes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS);
}

} // namespace neml2
