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

#include <c10/util/ArrayRef.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/Scalar.h>

namespace at
{
class Tensor;
} // namespace at

namespace neml2
{
using ATensor = at::Tensor;
using CScalar = c10::Scalar;

/// Fixed width dtypes (mirroring the definition in <torch/csrc/api/include/torch/types.h>)
// torch also provide unsigned integer types after version 2.2.1, but for backward compatibility
// reasons we don't include them here
constexpr auto kInt8 = c10::kChar;
constexpr auto kInt16 = c10::kShort;
constexpr auto kInt32 = c10::kInt;
constexpr auto kInt64 = c10::kLong;
constexpr auto kFloat16 = c10::kHalf;
constexpr auto kFloat32 = c10::kFloat;
constexpr auto kFloat64 = c10::kDouble;

// Device types (mirroring the definition in <c10/core/DeviceType.h>)
constexpr auto kCPU = c10::DeviceType::CPU;
constexpr auto kCUDA = c10::DeviceType::CUDA;

template <typename T, unsigned N>
using SmallVector = c10::SmallVector<T, N>;
template <typename T>
using ArrayRef = c10::ArrayRef<T>;
using TensorOptions = c10::TensorOptions;
using Dtype = c10::ScalarType;
using DeviceIndex = c10::DeviceIndex;
using Device = c10::Device;

using Size = int64_t;
using TensorShape = c10::SmallVector<Size, 8>;
using TensorShapeRef = c10::ArrayRef<Size>;

// Machine precision depending on the dtype
double machine_precision(Dtype);

/**
 * @brief Role in a function definition
 *
 * NONE is the default value,
 * INPUT stands for input variable,
 * OUTPUT stands for output variable,
 * PARAMETER stands for parameter (could request AD),
 * BUFFER stands for buffer.
 */
enum class FType : int8_t
{
  NONE = 0,
  INPUT = 1 << 0,
  OUTPUT = 1 << 1,
  PARAMETER = 1 << 2,
  BUFFER = 1 << 3
};
std::ostream & operator<<(std::ostream & os, FType f);

///@{ Constants
constexpr auto eps = std::numeric_limits<double>::epsilon();
constexpr double sqrt2 = 1.4142135623730951;
constexpr double invsqrt2 = 0.7071067811865475;
constexpr double pi = M_PI;
///@}
} // namespace neml2
