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

#include "python/neml2/tensors/TensorBase.h"
#include "python/neml2/tensors/DynamicView.h"
#include "python/neml2/tensors/IntmdView.h"
#include "python/neml2/tensors/BaseView.h"
#include "python/neml2/tensors/BatchView.h"
#include "python/neml2/tensors/StaticView.h"

template <class T>
void
def_TensorBase(pybind11::module_ & m, const std::string & type)
{
  def_DynamicView<T>(m, (type + "DynamicView").c_str());
  def_IntmdView<T>(m, (type + "IntmdView").c_str());
  def_BaseView<T>(m, (type + "BaseView").c_str());
  def_BatchView<T>(m, (type + "BatchView").c_str());
  def_StaticView<T>(m, (type + "StaticView").c_str());
}

// Explicit template instantiations
using namespace neml2;
#define INSTANTIATE_TENSORBASE(T)                                                                  \
  template void def_TensorBase<T>(pybind11::module_ &, const std::string &)
FOR_ALL_TENSORBASE(INSTANTIATE_TENSORBASE);
