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

#include <torch/types.h>

namespace neml2
{
using Real = double;
using Integer = int;
using Size = int64_t;
using TensorShape = torch::SmallVector<Size>;
using TensorShapeRef = torch::IntArrayRef;

// Bring in torch::indexing
namespace indexing
{
using namespace torch::indexing;
using TensorIndices = torch::SmallVector<TensorIndex>;
using TensorIndicesRef = torch::ArrayRef<TensorIndex>;
}

/**
 * @name RAII style default tensor options
 *
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

/// @name Default tolerances
///@{
/// Machine precision
Real & machine_precision();
/// The tolerance used in various algorithms
Real & tolerance();
/// A tighter tolerance used in various algorithms
Real & tighter_tolerance();
///@}

/// Default nested buffer name separator
std::string & buffer_name_separator();
/// Default nested parameter name separator
std::string & parameter_name_separator();

/**
 * A model can be _implicit. An implicit model need to be "solved": the state variables should be
 * iteratively updated until the residual becomes zero. During the solve, we only need derivatives
 * with respect to the input state. Therefore, the model can/should avoid unnecessary computations
 * by examining whether the current evaluation is part of the solve.
 */
bool & currently_solving_nonlinear_system();
} // namespace neml2
