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

#include "neml2/base/TensorName.h"
#include "neml2/base/Factory.h"
#include "neml2/base/Parser.h"
#include "neml2/base/LabeledAxisAccessor.h"

namespace neml2
{

template <typename T>
const T &
TensorName<T>::resolve() const
{
  // Try to parse as a number
  if (_value.defined())
    return _value;
  Real val;
  auto success = utils::parse_(val, _raw_str);
  if (success)
    return _value = resolve_number(val);

  // Try to parse as a tensor object
  if (_tensor)
    return *_tensor;
  try
  {
    _tensor = &Factory::get_object<T>("Tensors", _raw_str);
    return *_tensor;
  }
  catch (const FactoryException & err_tensor)
  {
    throw ParserException(
        "Failed to resolve tensor name '" + _raw_str + "'. Two attempts were made:" +
        "\n  1. Parsing it as a plain numeric literal failed with error message: " +
        utils::parse_failure_message<Real>(_raw_str) +
        "\n  2. Parsing it as a tensor object failed with error message: " + err_tensor.what());
  }
}

template <typename T>
T
TensorName<T>::resolve_number(Real val) const
{
  if constexpr (std::is_same_v<T, Tensor> || std::is_same_v<T, ATensor>)
    return Scalar::create(val);
  else
    return T::full(val);
}

// Explicit instantiations
template struct TensorName<ATensor>;
#define INSTANTIATE(T) template struct TensorName<T>;
FOR_ALL_TENSORBASE(INSTANTIATE);
} // namesace neml2
