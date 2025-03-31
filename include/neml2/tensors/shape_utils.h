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

#include <vector>
#include "neml2/tensors/indexing.h"
#include "neml2/misc/errors.h"

namespace neml2::utils
{
/**
 * Two tensors are said to be broadcastable if
 * 1. Base shapes are the same
 * 2. Batch shapes are broadcastable (see sizes_broadcastable)
 */
template <class... T>
bool broadcastable(const T &... tensors);

/**
 * Test if the tensors are batch-broadcastable.
 * @see sizes_broadcastable
 */
template <class... T>
bool batch_broadcastable(const T &... tensors);

/**
 * Test if the tensors are base-broadcastable.
 * @see sizes_broadcastable
 */
template <class... T>
bool base_broadcastable(const T &... tensors);

/**
 * @brief The batch dimension after broadcasting
 *
 * This should be as simple as the maximum batch_dim() among all arguments.
 */
template <class... T>
Size broadcast_batch_dim(const T &...);

/// Check if all shapes are the *same*.
template <class... T>
bool sizes_same(T &&... shapes);

/**
 * @brief Check if the shapes are broadcastable.
 *
 * Shapes are said to be broadcastable if, starting from the trailing dimension and
 * iterating backward, the dimension sizes either are equal, one of them is 1, or one of them does
 * not exist.
 */
template <class... T>
bool sizes_broadcastable(const T &... shapes);

/**
 * @brief Return the broadcast shape of all the shapes.
 */
template <class... T>
TensorShape broadcast_sizes(const T &... shapes);

/**
 * @brief The flattened storage size of a tensor with given shape
 *
 * For example,
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cpp
 * storage_size({}) == 1;
 * storage_size({0}) == 0;
 * storage_size({1}) == 1;
 * storage_size({1, 2, 3}) == 6;
 * storage_size({5, 1, 1}) == 5;
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
Size storage_size(TensorShapeRef shape);

template <typename... S>
TensorShape add_shapes(const S &...);

/**
 * @brief Pad shape \p s to dimension \p dim by prepending sizes of \p pad.
 *
 * @param s The original shape to pad
 * @param dim The resulting dimension
 * @param pad The values used to pad the shape, default to 1
 * @return TensorShape The padded shape with dimension \p dim
 */
TensorShape pad_prepend(TensorShapeRef s, Size dim, Size pad = 1);

namespace details
{
template <typename... S>
TensorShape add_shapes_impl(TensorShape &, TensorShapeRef, const S &...);
} // namespace details
} // namespace neml2::utils

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

namespace neml2::utils
{
template <class... T>
bool
broadcastable(const T &... tensors)
{
  if (!sizes_same(tensors.base_sizes()...))
    return false;
  return batch_broadcastable(tensors...);
}

template <class... T>
bool
batch_broadcastable(const T &... tensors)
{
  return sizes_broadcastable(tensors.batch_sizes().concrete()...);
}

template <class... T>
bool
base_broadcastable(const T &... tensors)
{
  return sizes_broadcastable(tensors.base_sizes()...);
}

template <class... T>
Size
broadcast_batch_dim(const T &... tensor)
{
  return std::max({tensor.batch_dim()...});
}

template <class... T>
bool
sizes_same(T &&... shapes)
{
  auto all_shapes = std::vector<TensorShapeRef>{std::forward<T>(shapes)...};
  for (size_t i = 0; i < all_shapes.size() - 1; i++)
    if (all_shapes[i] != all_shapes[i + 1])
      return false;
  return true;
}

template <class... T>
bool
sizes_broadcastable(const T &... shapes)
{
  auto dim = std::max({shapes.size()...});
  auto all_shapes_padded = std::vector<TensorShape>{pad_prepend(shapes, dim)...};

  for (size_t i = 0; i < dim; i++)
  {
    Size max_sz = 1;
    for (const auto & s : all_shapes_padded)
    {
      if (max_sz == 1)
      {
#ifndef NDEBUG
        if (s[i] <= 0)
          throw NEMLException("Found a size equal or less than 0: " + std::to_string(s[i]));
#endif
        if (s[i] > 1)
          max_sz = s[i];
      }
      else if (s[i] != 1 && s[i] != max_sz)
        return false;
    }
  }

  return true;
}

template <class... T>
TensorShape
broadcast_sizes(const T &... shapes)
{
#ifndef NDEBUG
  if (!sizes_broadcastable(shapes...))
    throw NEMLException("Shapes not broadcastable");
#endif

  auto dim = std::max({shapes.size()...});
  auto all_shapes_padded = std::vector<TensorShape>{pad_prepend(shapes, dim)...};
  auto bshape = TensorShape(dim, 1);

  for (size_t i = 0; i < dim; i++)
    for (const auto & s : all_shapes_padded)
      if (s[i] > bshape[i])
        bshape[i] = s[i];

  return bshape;
}

template <typename... S>
TensorShape
add_shapes(const S &... shape)
{
  TensorShape net;
  return details::add_shapes_impl(net, shape...);
}

namespace details
{
template <typename... S>
TensorShape
add_shapes_impl(TensorShape & net, TensorShapeRef s, const S &... rest)
{
  net.insert(net.end(), s.begin(), s.end());

  if constexpr (sizeof...(rest) == 0)
    return std::move(net);
  else
    return add_shapes_impl(net, rest...);
}
} // namespace details
} // namespace neml2::utils
