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
#include "neml2/misc/errors.h"

namespace neml2::utils
{

/**
 * @brief Sum all elements in an array.
 *
 * @note This assumes, of course, that the + operator is defined for type T.
 */
template <typename T, std::size_t N>
static T sum_array(const std::array<T, N> & arr);

/**
 * @brief Helper function to normalize a dimension index to be non-negative given the lower- and
 * upper-bound of the context.
 *
 * @param d The dimension index to normalize
 * @param dl The lower-bound (inclusive)
 * @param du The upper-bound (exclusive)
 * @return Size The normalized dimension index
 */
Size normalize_dim(Size d, Size dl, Size du);

/**
 * @brief Helper function to normalize multiple dimension indices to be non-negative given the
 * lower- and upper-bound of the context.
 *
 * @param d The dimension indices to normalize
 * @param dl The lower-bound (inclusive)
 * @param du The upper-bound (exclusive)
 * @return TensorShape The normalized dimension indices
 */
TensorShape normalize_dims(ArrayRef<Size> d, Size dl, Size du);

/**
 * @brief Helper function to normalize a iterator-like index to be non-negative given the lower- and
 * upper-bound of the context.
 *
 * @param d The iterator index to normalize
 * @param dl The lower-bound (inclusive)
 * @param du The upper-bound (exclusive)
 * @return Size The normalized iterator index
 */
Size normalize_itr(Size d, Size dl, Size du);

/**
 * @brief Helper function to normalize multiple iterator-like indices to be non-negative given the
 * lower- and upper-bound of the context.
 *
 * @param d The iterator indices to normalize
 * @param dl The lower-bound (inclusive)
 * @param du The upper-bound (exclusive)
 * @return TensorShape The normalized iterator indices
 */
TensorShape normalize_itrs(ArrayRef<Size> d, Size dl, Size du);

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
 * Two tensors are said to be broadcastable if
 * 1. Base shapes are broadcastable
 * 2. Intermediate shapes are broadcastable
 * 3. Dynamic shapes are broadcastable
 */
template <class... T>
bool broadcastable(const T &... tensors);

/**
 * Test if the tensors are dynamic-broadcastable.
 * @see sizes_broadcastable
 */
template <class... T>
bool dynamic_broadcastable(const T &... tensors);

/**
 * Test if the tensors are intermediate-broadcastable.
 * @see sizes_broadcastable
 */
template <class... T>
bool intmd_broadcastable(const T &... tensors);

/**
 * Test if the tensors are base-broadcastable.
 * @see sizes_broadcastable
 */
template <class... T>
bool base_broadcastable(const T &... tensors);

/// The dynamic dimension after broadcasting
template <class... T>
Size broadcast_dynamic_dim(const T &...);

/// The intermediate dimension after broadcasting
template <class... T>
Size broadcast_intmd_dim(const T &...);

/// The base dimension after broadcasting
template <class... T>
Size broadcast_base_dim(const T &...);

/**
 * @brief Return the broadcast shape of all the shapes.
 */
template <class... T>
TensorShape broadcast_sizes(const T &... shapes);

/**
 * @brief Number of elements in a tensor with given shape
 *
 * For example,
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cpp
 * numel({}) == 1;
 * numel({0}) == 0;
 * numel({1}) == 1;
 * numel({1, 2, 3}) == 6;
 * numel({5, 1, 1}) == 5;
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
Size numel(TensorShapeRef shape);

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
TensorShape pad_prepend(TensorShapeRef s, std::size_t dim, Size pad = 1);

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
template <typename T, std::size_t N>
static T
sum_array(const std::array<T, N> & arr)
{
  return std::accumulate(arr.begin(), arr.end(), T(0), [](T sum, T x) { return sum + x; });
}

template <class... T>
bool
sizes_broadcastable(const T &... shapes)
{
  auto dim = std::max({shapes.size()...});
  auto all_shapes_padded = std::vector<TensorShape>{pad_prepend(shapes, dim)...};

  for (std::size_t i = 0; i < dim; i++)
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
bool
broadcastable(const T &... tensors)
{
  return dynamic_broadcastable(tensors...) && intmd_broadcastable(tensors...) &&
         base_broadcastable(tensors...);
}

template <class... T>
bool
dynamic_broadcastable(const T &... tensors)
{
  return sizes_broadcastable(tensors.dynamic_sizes().concrete()...);
}

template <class... T>
bool
intmd_broadcastable(const T &... tensors)
{
  return sizes_broadcastable(tensors.intmd_sizes()...);
}

template <class... T>
bool
base_broadcastable(const T &... tensors)
{
  return sizes_broadcastable(tensors.base_sizes()...);
}

template <class... T>
Size
broadcast_dynamic_dim(const T &... tensor)
{
  return std::max({tensor.dynamic_dim()...});
}

template <class... T>
Size
broadcast_intmd_dim(const T &... tensor)
{
  return std::max({tensor.intmd_dim()...});
}

template <class... T>
Size
broadcast_base_dim(const T &... tensor)
{
  return std::max({tensor.base_dim()...});
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

  for (std::size_t i = 0; i < dim; i++)
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
