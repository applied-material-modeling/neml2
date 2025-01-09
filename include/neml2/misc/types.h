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

#include <cstdint>
#include <iosfwd>

namespace c10
{
template <typename T, unsigned N>
class SmallVector;
template <typename T>
class ArrayRef;
struct TensorOptions;
enum class ScalarType : int8_t;
struct Device;
} // namespace c10

namespace at
{
class Tensor;
using ScalarType = c10::ScalarType;
namespace indexing
{
struct Slice;
struct TensorIndex;
}
} // namespace at

namespace torch
{
using Tensor = at::Tensor;
template <typename T, unsigned N>
using SmallVector = c10::SmallVector<T, N>;
template <typename T>
using ArrayRef = c10::ArrayRef<T>;
using TensorOptions = c10::TensorOptions;
using Dtype = at::ScalarType;
using Device = c10::Device;
} // namespace torch

namespace neml2
{
using Real = double;
using Size = int64_t;
using Integer = int64_t;
using TensorShape = torch::SmallVector<Size, 8>;
using TensorShapeRef = torch::ArrayRef<Size>;

namespace indexing
{
using namespace at::indexing;
using TensorIndices = torch::SmallVector<TensorIndex, 8>;
using TensorIndicesRef = torch::ArrayRef<TensorIndex>;
} // namespace indexing

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
} // namespace neml2
