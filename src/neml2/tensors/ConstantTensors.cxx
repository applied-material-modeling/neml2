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

#include "neml2/tensors/ConstantTensors.h"

namespace neml2
{
ConstantTensors::ConstantTensors()
{
  const auto DITO = default_integer_tensor_options();

  _full_to_mandel_map = Tensor::create({0, 4, 8, 5, 2, 1}, DITO);
  _mandel_to_full_map = Tensor::create({0, 5, 4, 5, 1, 3, 4, 3, 2}, DITO);
  _full_to_mandel_factor = Tensor::create({1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2});
  _mandel_to_full_factor =
      Tensor::create({1.0, invsqrt2, invsqrt2, invsqrt2, 1.0, invsqrt2, invsqrt2, invsqrt2, 1.0});

  _full_to_skew_map = Tensor::create({7, 2, 3}, DITO);
  _skew_to_full_map = Tensor::create({0, 2, 1, 2, 0, 0, 1, 0, 0}, DITO);
  _full_to_skew_factor = Tensor::create({1.0, 1.0, 1.0});
  _skew_to_full_factor = Tensor::create({0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0});
}

ConstantTensors &
ConstantTensors::get()
{
  static ConstantTensors cts;
  return cts;
}

const Tensor &
ConstantTensors::full_to_mandel_map()
{
  return get()._full_to_mandel_map;
}

const Tensor &
ConstantTensors::mandel_to_full_map()
{
  return get()._mandel_to_full_map;
}

const Tensor &
ConstantTensors::full_to_mandel_factor()
{
  return get()._full_to_mandel_factor;
}

const Tensor &
ConstantTensors::mandel_to_full_factor()
{
  return get()._mandel_to_full_factor;
}

const Tensor &
ConstantTensors::full_to_skew_map()
{
  return get()._full_to_skew_map;
}

const Tensor &
ConstantTensors::skew_to_full_map()
{
  return get()._skew_to_full_map;
}

const Tensor &
ConstantTensors::full_to_skew_factor()
{
  return get()._full_to_skew_factor;
}

const Tensor &
ConstantTensors::skew_to_full_factor()
{
  return get()._skew_to_full_factor;
}
}
