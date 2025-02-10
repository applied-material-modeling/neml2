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

#include "neml2/tensors/Tensor.h"

namespace neml2
{
/**
 * @brief A helper class to hold static data of type ATensor
 *
 * This class exists because ATensor cannot be declared as constexpr nor as static data in the
 * global scope. The former is obvious. The latter is because at the time static variables are
 * initialized, some torch data structures have not been properly initialized yet.
 *
 */
struct ConstantTensors
{
  ConstantTensors();

  // Get the global constants
  static ConstantTensors & get();

  static const Tensor & full_to_mandel_map();
  static const Tensor & mandel_to_full_map();
  static const Tensor & full_to_mandel_factor();
  static const Tensor & mandel_to_full_factor();
  static const Tensor & full_to_skew_map();
  static const Tensor & skew_to_full_map();
  static const Tensor & full_to_skew_factor();
  static const Tensor & skew_to_full_factor();

private:
  Tensor _full_to_mandel_map;
  Tensor _mandel_to_full_map;
  Tensor _full_to_mandel_factor;
  Tensor _mandel_to_full_factor;
  Tensor _full_to_skew_map;
  Tensor _skew_to_full_map;
  Tensor _full_to_skew_factor;
  Tensor _skew_to_full_factor;
};
}
