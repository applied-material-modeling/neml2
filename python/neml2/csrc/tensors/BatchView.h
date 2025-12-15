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
void def_BatchView(pybind11::module_ & m, const std::string & name);

/**
 * @brief Convenient shim for working with batch (dynamic+intmd) dimensions
 *
 * The view does NOT extend the life of of the wrapped tensor.
 */
template <class T>
class BatchView
{
public:
  BatchView(T * data);

  // These methods mirror TensorBase (the batch_xxx ones)
  neml2::Size dim() const;
  neml2::TensorShape sizes() const;
  neml2::Size size(neml2::Size) const;
  T index(const neml2::indexing::TensorIndices &) const;
  T slice(neml2::Size, const neml2::indexing::Slice &) const;
  void index_put_(const neml2::indexing::TensorIndices &, const neml2::ATensor &);
  T expand(neml2::TensorShapeRef, neml2::TensorShapeRef) const;
  T expand_as(const neml2::Tensor &) const;
  T reshape(neml2::TensorShapeRef, neml2::TensorShapeRef) const;
  T squeeze(neml2::Size) const;
  T unsqueeze(neml2::Size, neml2::Size n = 1) const;
  T transpose(neml2::Size d1, neml2::Size d2) const;
  T movedim(neml2::Size, neml2::Size) const;
  T flatten() const;

private:
  T * _data;
};

#define EXPORT_BATCHVIEW(T)                                                                        \
  extern template void def_BatchView<neml2::T>(pybind11::module_ &, const std::string &);          \
  extern template class BatchView<neml2::T>
FOR_ALL_TENSORBASE(EXPORT_BATCHVIEW);
#undef EXPORT_BATCHVIEW
