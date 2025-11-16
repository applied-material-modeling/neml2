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

#include <pybind11/pytypes.h>

#include "python/neml2/tensors/PrimitiveTensor.h"

namespace py = pybind11;
using namespace neml2;

namespace detail
{
template <class T, std::size_t N, std::size_t... Is>
T
call_fill_impl(const std::array<Scalar, N> & vals, std::index_sequence<Is...>)
{
  // Expands to: T::fill(vals[0], ..., vals[N-1])
  return T::fill(vals[Is]...);
}

template <class T, std::size_t N>
T
call_fill(const std::array<Scalar, N> & vals)
{
  return call_fill_impl<T, N>(vals, std::make_index_sequence<N>{});
}
} // namespace detail

template <class T>
void
def_PrimitiveTensor(py::module_ & m, const std::string & type)
{
  auto pyc = m.attr(type.c_str());
  auto c = py::class_<T>(pyc);
  c.def(py::init<const ATensor &, Size>(), py::arg("tensor"), py::arg("intmd_dim") = 0);

  // Static methods
  c.def_static(
       "empty",
       [](NEML2_TENSOR_OPTIONS_VARGS) { return T::empty(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "empty",
          [](TensorShapeRef dynamic_sizes, TensorShapeRef intmd_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::empty(dynamic_sizes, intmd_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes") = TensorShapeRef{},
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return T::zeros(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](TensorShapeRef dynamic_sizes, TensorShapeRef intmd_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::zeros(dynamic_sizes, intmd_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes") = TensorShapeRef{},
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return T::ones(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](TensorShapeRef dynamic_sizes, TensorShapeRef intmd_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::ones(dynamic_sizes, intmd_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes") = TensorShapeRef{},
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](double init, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::full(init, NEML2_TENSOR_OPTIONS); },
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](TensorShapeRef dynamic_sizes,
             TensorShapeRef intmd_sizes,
             double init,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return T::full(dynamic_sizes, intmd_sizes, init, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "rand",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return T::rand(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "random",
          [](TensorShapeRef dynamic_sizes, TensorShapeRef intmd_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::rand(dynamic_sizes, intmd_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS);

  // fill -- this is a fun one!
  c.def_static(
      "fill",
      [](const py::args & args, const py::kwargs & kwargs)
      {
        constexpr auto N = T::const_base_numel;
        if (args.size() != N)
          throw py::type_error("Expected " + std::to_string(N) + " arguments, got " +
                               std::to_string(args.size()));

        // arguments must be all Scalars or all numbers
        bool all_scalar = true;
        bool all_numbers = true;
        for (std::size_t i = 0; i < N; ++i)
        {
          if (!py::isinstance<Scalar>(args[i]))
            all_scalar = false;
          if (!py::isinstance<double>(args[i]))
            all_numbers = false;
        }
        if (!all_scalar && !all_numbers)
          throw py::type_error("All arguments must be either neml2.tensors.Scalar or float.");
        if (all_scalar && !kwargs.empty())
          throw py::type_error(
              "When passing neml2.tensors.Scalar as arguments, no keyword arguments are allowed.");

        // get options from kwargs
        TensorOptions options = default_tensor_options();
        if (kwargs.contains("dtype"))
          options = options.dtype(kwargs["dtype"].cast<Dtype>());
        if (kwargs.contains("device"))
          options = options.device(kwargs["device"].cast<Device>());
        if (kwargs.contains("requires_grad"))
          options = options.requires_grad(kwargs["requires_grad"].cast<bool>());

        // convert first N positional arguments to Scalars
        std::array<Scalar, N> vals;
        for (std::size_t i = 0; i < N; ++i)
        {
          if (py::isinstance<Scalar>(args[i]))
            vals[i] = args[i].cast<Scalar>();
          else if (py::isinstance<double>(args[i]))
            vals[i] = Scalar(args[i].cast<double>(), options);
        }

        // call the implementation
        return detail::call_fill<T>(vals);
      });
}

// Explicit template instantiations
#define INSTANTIATE_PRIMITIVETENSOR(T)                                                             \
  template void def_PrimitiveTensor<T>(py::module_ &, const std::string &)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_PRIMITIVETENSOR);
