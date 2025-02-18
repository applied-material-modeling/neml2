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

#include "neml2/models/map_types.h"

namespace neml2
{
/// @brief  Concatenate the tensors in the ValueMap along the batch dimension
/// @param results The results to concatenate
/// @param batch_dim The batch dimension along which to concatenate
/// @return ValueMap with the tensors concatenated along the batch dimension
ValueMap valuemap_cat_reduce(std::vector<ValueMap> && results, Size batch_dim);

/// @brief Move all tensors in a ValueMap to a device
/// @param x input ValueMap
/// @param device target device
/// @return ValueMap with all tensors moved
ValueMap valuemap_move_device(ValueMap && x, Device device);

/// @brief No operation
ValueMap valuemap_no_operation(ValueMap && x);
}
