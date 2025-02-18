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

#include "neml2/misc/assertions.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{
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

/**
 * @brief A helper function to assert that all tensors are base-broadcastable
 *
 * In most cases, this assertion is necessary as libTorch will raise runtime_errors if things go
 * wrong. Therefore, this function is just so that we can detect errors before libTorch does and
 * emit some more mearningful error messages within the NEML2 context.
 */
template <class... T>
void neml_assert_base_broadcastable(const T &...);

/**
 * @brief A helper function to assert that (in Debug mode) all tensors are base-broadcastable
 *
 * In most cases, this assertion is necessary as libTorch will raise runtime_errors if things go
 * wrong. Therefore, this function is just so that we can detect errors before libTorch does and
 * emit some more mearningful error messages within the NEML2 context.
 */
template <class... T>
void neml_assert_base_broadcastable_dbg(const T &...);
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
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
  neml_assert_dbg(utils::broadcastable(tensors...),
                  "The ",
                  sizeof...(tensors),
                  " operands are not broadcastable. The batch shapes are ",
                  tensors.batch_sizes()...,
                  ", and the base shapes are ",
                  tensors.base_sizes()...);
}

template <class... T>
void
neml_assert_batch_broadcastable(const T &... tensors)
{
  neml_assert(utils::batch_broadcastable(tensors...),
              "The ",
              sizeof...(tensors),
              " operands are not batch-broadcastable. The batch shapes are ",
              tensors.batch_sizes()...);
}

template <class... T>
void
neml_assert_batch_broadcastable_dbg([[maybe_unused]] const T &... tensors)
{
  neml_assert_dbg(utils::batch_broadcastable(tensors...),
                  "The ",
                  sizeof...(tensors),
                  " operands are not batch-broadcastable. The batch shapes are ",
                  tensors.batch_sizes()...);
}

template <class... T>
void
neml_assert_base_broadcastable(const T &... tensors)
{
  neml_assert(utils::base_broadcastable(tensors...),
              "The ",
              sizeof...(tensors),
              " operands are not base-broadcastable. The base shapes are ",
              tensors.base_sizes()...);
}

template <class... T>
void
neml_assert_base_broadcastable_dbg([[maybe_unused]] const T &... tensors)
{
  neml_assert_dbg(utils::base_broadcastable(tensors...),
                  "The ",
                  sizeof...(tensors),
                  " operands are not base-broadcastable. The base shapes are ",
                  tensors.base_sizes()...);
}
} // namespace neml2
