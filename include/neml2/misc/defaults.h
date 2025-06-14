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

#define NEML2_TORCH_ENUM(type) k##type
#define NEML2_STRINGIFY(x) #x
#define NEML2_DEFAULT_INTEGER_DTYPE Int64
#define NEML2_DEFAULT_INTEGER_DTYPE_STR NEML2_STRINGIFY(Int64)
#define NEML2_DEFAULT_INTEGER_DTYPE_ENUM NEML2_TORCH_ENUM(Int64)

namespace neml2
{
/**
 * @name RAII style default tensor options
 *
 * The factory methods like `at::arange`, `at::ones`, `at::zeros`, `at::rand` etc.
 * accept a common argument to configure the properties of the tensor being created. We predefine
 * a default tensor configuration in NEML2. This default configuration is consistently used
 * throughout NEML2.
 *
 * See https://pytorch.org/cppdocs/notes/tensor_creation.html#configuring-properties-of-the-tensor
 * for more details.
 */
///@{
/// Set default dtype
void set_default_dtype(Dtype dtype);
/// Get default dtype
Dtype get_default_dtype();
/// Default floating point tensor options
TensorOptions default_tensor_options();
/// Set default integer dtype
void set_default_integer_dtype(Dtype dtype);
/// Get default integer dtype
Dtype get_default_integer_dtype();
/// Default integral tensor options
TensorOptions default_integer_tensor_options();
///@}

namespace details
{
/// Default integral type
Dtype & default_integer_dtype();
}
} // namespace neml2
