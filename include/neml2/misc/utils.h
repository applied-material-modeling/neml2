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
#include "neml2/misc/error.h"

namespace neml2
{

/**
 * Two tensors are said to be broadcastable if
 * 1. Base shapes are the same
 * 2. Batch shapes are broadcastable (see sizes_broadcastable)
 */
template <class... T>
bool broadcastable(const T &... tensors);

/**
 * @brief The batch dimension after broadcasting
 *
 * This should be as simple as the maximum batch_dim() among all arguments.
 */
template <class... T>
Size broadcast_batch_dim(const T &...);

/**
 * @brief A helper function to assert that all tensors are broadcastable
 *
 * In most cases, this assertion is necessary as libTorch will raise runtime_errors if things go
 * wrong. Therefore, this function is just so that we can detect errors before libTorch does and
 * emit some more mearningful error messages within the NEML2 context.
 */
template <class... T>
void neml_assert_broadcastable(const T &...);

/**
 * @brief A helper function to assert (in Debug mode) that all tensors are broadcastable
 *
 * In most cases, this assertion is necessary as libTorch will raise runtime_errors if things go
 * wrong. Therefore, this function is just so that we can detect errors before libTorch does and
 * emit some more mearningful error messages within the NEML2 context.
 */
template <class... T>
void neml_assert_broadcastable_dbg(const T &...);

/**
 * @brief A helper function to assert that all tensors are batch-broadcastable
 *
 * In most cases, this assertion is necessary as libTorch will raise runtime_errors if things go
 * wrong. Therefore, this function is just so that we can detect errors before libTorch does and
 * emit some more mearningful error messages within the NEML2 context.
 */
template <class... T>
void neml_assert_batch_broadcastable(const T &...);

/**
 * @brief A helper function to assert that (in Debug mode) all tensors are batch-broadcastable
 *
 * In most cases, this assertion is necessary as libTorch will raise runtime_errors if things go
 * wrong. Therefore, this function is just so that we can detect errors before libTorch does and
 * emit some more mearningful error messages within the NEML2 context.
 */
template <class... T>
void neml_assert_batch_broadcastable_dbg(const T &...);

namespace utils
{
/// Demangle a piece of cxx abi type information
std::string demangle(const char * name);

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

/// @brief Extract the batch shape of a tensor given batch dimension
/// The extracted batch shape will be _traceable_. @see neml2::TraceableTensorShape
TraceableTensorShape extract_batch_sizes(const torch::Tensor & tensor, Size batch_dim);

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

template <typename... S>
TraceableTensorShape add_traceable_shapes(const S &... shape);

/**
 * @brief Pad shape \p s to dimension \p dim by prepending sizes of \p pad.
 *
 * @param s The original shape to pad
 * @param dim The resulting dimension
 * @param pad The values used to pad the shape, default to 1
 * @return TensorShape The padded shape with dimension \p dim
 */
TensorShape pad_prepend(TensorShapeRef s, Size dim, Size pad = 1);
torch::Tensor pad_prepend(const torch::Tensor & s, Size dim, Size pad = 1);

std::string indentation(int level, int indent = 2);

template <typename T>
std::string stringify(const T & t);

namespace details
{
template <typename... S>
TensorShape add_shapes_impl(TensorShape &, TensorShapeRef, const S &...);

template <typename... S>
TraceableTensorShape
add_traceable_shapes_impl(TraceableTensorShape &, const TraceableTensorShape &, const S &...);
} // namespace details
} // namespace utils
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
template <class... T>
bool
broadcastable(const T &... tensors)
{
  if (!utils::sizes_same(tensors.base_sizes()...))
    return false;
  return utils::sizes_broadcastable(tensors.batch_sizes().concrete()...);
}

template <class... T>
Size
broadcast_batch_dim(const T &... tensor)
{
  return std::max({tensor.batch_dim()...});
}

template <class... T>
void
neml_assert_broadcastable(const T &... tensors)
{
  neml_assert(broadcastable(tensors...),
              "The ",
              sizeof...(tensors),
              " operands are not broadcastable. The batch shapes are ",
              tensors.batch_sizes()...,
              ", and the base shapes are ",
              tensors.base_sizes()...);
}

template <class... T>
void
neml_assert_broadcastable_dbg([[maybe_unused]] const T &... tensors)
{
#ifndef NDEBUG
  neml_assert_dbg(broadcastable(tensors...),
                  "The ",
                  sizeof...(tensors),
                  " operands are not broadcastable. The batch shapes are ",
                  tensors.batch_sizes()...,
                  ", and the base shapes are ",
                  tensors.base_sizes()...);
#endif
}

template <class... T>
void
neml_assert_batch_broadcastable(const T &... tensors)
{
  neml_assert(utils::sizes_broadcastable(tensors.batch_sizes().concrete()...),
              "The ",
              sizeof...(tensors),
              " operands are not batch-broadcastable. The batch shapes are ",
              tensors.batch_sizes()...);
}

template <class... T>
void
neml_assert_batch_broadcastable_dbg([[maybe_unused]] const T &... tensors)
{
#ifndef NDEBUG
  neml_assert_dbg(utils::sizes_broadcastable(tensors.batch_sizes().concrete()...),
                  "The ",
                  sizeof...(tensors),
                  " operands are not batch-broadcastable. The batch shapes are ",
                  tensors.batch_sizes()...);
#endif
}

namespace utils
{
template <class... T>
bool
sizes_same(T &&... shapes)
{
  auto all_shapes = std::vector<TensorShapeRef>{shapes...};
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
        neml_assert_dbg(s[i] > 0, "Found a size equal or less than 0.");
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
  neml_assert_dbg(sizes_broadcastable(shapes...), "Shapes not broadcastable: ", shapes...);

  auto dim = std::max({shapes.size()...});
  auto all_shapes_padded = std::vector<TensorShape>{pad_prepend(shapes, dim)...};
  auto bshape = TensorShape(dim, 1);

  for (size_t i = 0; i < dim; i++)
    for (const auto & s : all_shapes_padded)
      if (s[i] > bshape[i])
        bshape[i] = s[i];

  return bshape;
}

// The directives below suppress the warning about the variables being used uninitialized
// I believe this is a false positive generated by gcc.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

template <typename... S>
TensorShape
add_shapes(const S &... shape)
{
  TensorShape net;
  return details::add_shapes_impl(net, shape...);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <typename... S>
TraceableTensorShape
add_traceable_shapes(const S &... shape)
{
  TraceableTensorShape net;
  return details::add_traceable_shapes_impl(net, shape...);
}

template <typename T>
std::string
stringify(const T & t)
{
  std::ostringstream os;
  os << t;
  return os.str();
}

template <>
inline std::string
stringify(const bool & t)
{
  return t ? "true" : "false";
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

template <typename... S>
TraceableTensorShape
add_traceable_shapes_impl(TraceableTensorShape & net,
                          const TraceableTensorShape & s,
                          const S &... rest)
{
  net.insert(net.end(), s.begin(), s.end());

  if constexpr (sizeof...(rest) == 0)
    return std::move(net);
  else
    return add_traceable_shapes_impl(net, rest...);
}
} // namespace details
} // namespace utils
} // namespace neml2
