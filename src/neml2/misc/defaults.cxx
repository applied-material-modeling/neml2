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

#include "neml2/misc/defaults.h"

namespace neml2
{
void
set_default_dtype(Dtype dtype)
{
  c10::set_default_dtype(scalarTypeToTypeMeta(dtype));
}

Dtype
get_default_dtype()
{
  return c10::get_default_dtype_as_scalartype();
}

TensorOptions
default_tensor_options()
{
  return TensorOptions().dtype(c10::get_default_dtype());
}

void
set_default_integer_dtype(Dtype dtype)
{
  details::default_integer_dtype() = dtype;
}

Dtype
get_default_integer_dtype()
{
  return details::default_integer_dtype();
}

TensorOptions
default_integer_tensor_options()
{
  return TensorOptions().dtype(get_default_integer_dtype());
}

namespace details
{
Dtype &
default_integer_dtype()
{
  static Dtype _default_integer_dtype = NEML2_DEFAULT_INTEGER_DTYPE_ENUM;
  return _default_integer_dtype;
}
} // namespace details
} // namespace neml2
