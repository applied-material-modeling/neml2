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

#include "neml2/misc/types.h"
#include "neml2/tensors/indexing.h"
#include "neml2/tensors/tensors.h"

#include <pybind11/pybind11.h>

template <class T>
void def_StaticView(pybind11::module_ & m, const std::string & name);

/**
 * @brief Convenient shim for working with static (intmd+base) dimensions
 *
 * The view does NOT extend the life of of the wrapped tensor.
 */
template <class T>
class StaticView
{
public:
  StaticView(T * data);

  // These methods mirror TensorBase (the dynamic_xxx ones)
  neml2::Size dim() const;
  neml2::TensorShapeRef sizes() const;
  neml2::Size size(neml2::Size) const;
  neml2::Tensor expand(neml2::TensorShapeRef, neml2::TensorShapeRef) const;
  neml2::Tensor expand_as(const neml2::Tensor &) const;
  neml2::Tensor reshape(neml2::TensorShapeRef, neml2::TensorShapeRef) const;
  neml2::Tensor flatten() const;

private:
  T * _data;
};

#define EXPORT_STATICVIEW(T)                                                                       \
  extern template void def_StaticView<neml2::T>(pybind11::module_ &, const std::string &);         \
  extern template class StaticView<neml2::T>
FOR_ALL_TENSORBASE(EXPORT_STATICVIEW);
#undef EXPORT_STATICVIEW
