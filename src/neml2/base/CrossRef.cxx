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

#include "neml2/base/CrossRef.h"
#include "neml2/base/Factory.h"
#include "neml2/misc/parser_utils.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <>
CrossRef<torch::Tensor>::operator torch::Tensor() const
{
  try
  {
    // If it is just a number, we can still create a tensor out of it
    return torch::tensor(utils::parse<Real>(_raw_str), default_tensor_options());
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another tensor
    return Factory::get_object<torch::Tensor>("Tensors", _raw_str);
  }
}

template <>
CrossRef<Tensor>::operator Tensor() const
{
  try
  {
    // If it is just a number, we can still create a Scalar out of it
    return Tensor::full({}, {}, utils::parse<Real>(_raw_str), default_tensor_options());
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another Tensor
    return Factory::get_object<Tensor>("Tensors", _raw_str);
  }
}

#define CROSSREF_SPECIALIZE_PRIMITIVETENSOR_IMPL(T)                                                \
  template <>                                                                                      \
  CrossRef<T>::operator T() const                                                                  \
  {                                                                                                \
    try                                                                                            \
    {                                                                                              \
      return T::full(utils::parse<Real>(_raw_str));                                                \
    }                                                                                              \
    catch (const ParserException & e)                                                              \
    {                                                                                              \
      return Factory::get_object<T>("Tensors", _raw_str);                                          \
    }                                                                                              \
  }                                                                                                \
  static_assert(true)
FOR_ALL_PRIMITIVETENSOR(CROSSREF_SPECIALIZE_PRIMITIVETENSOR_IMPL);
#undef CROSSREF_SPECIALIZE_PRIMITIVETENSOR_IMPL

#define INSTANTIATE_CROSSREF(T) template class CrossRef<T>
FOR_ALL_TENSORBASE(INSTANTIATE_CROSSREF);
#undef INSTANTIATE_CROSSREF
} // namesace neml2
