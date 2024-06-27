// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/base/config.h"

namespace neml2
{
typedef double Real;
typedef int Integer;
typedef int64_t TorchSize;
typedef std::vector<TorchSize> TorchShape;
typedef torch::IntArrayRef TorchShapeRef;
typedef at::indexing::TensorIndex TorchIndex;
typedef std::vector<at::indexing::TensorIndex> TorchSlice;

/**
 * The factory methods like `torch::arange`, `torch::ones`, `torch::zeros`, `torch::rand` etc.
 * accept a common argument to configure the properties of the tensor being created. We predefine
 * a default tensor configuration in NEML2. This default configuration is consistently used
 * throughout NEML2.
 *
 * See https://pytorch.org/cppdocs/notes/tensor_creation.html#configuring-properties-of-the-tensor
 * for more details.
 */
///@{
/// Default floating point tensor options
torch::TensorOptions & default_tensor_options();
/// Default integral tensor options
torch::TensorOptions & default_integer_tensor_options();
/// Default floating point type
torch::Dtype & default_dtype();
/// Default integral type
torch::Dtype & default_integer_dtype();
/// Default device
torch::Device & default_device();
///@}

/// Machine precision
// TODO: make this depend on the current dtype
Real & machine_precision();

/// The tolerance used in various algorithms
// TODO: make this depend on the current dtype
Real & tolerance();

/// A tighter tolerance used in various algorithms
// TODO: make this depend on the current dtype
Real & tighter_tolerance();

} // namespace neml2
