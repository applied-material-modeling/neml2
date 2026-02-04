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
/// Assert that all tensors are broadcastable
///@{
template <class... T>
void neml_assert_broadcastable(const T &...);

#ifndef NDEBUG
#define neml_assert_broadcastable_dbg(...) ::neml2::neml_assert_broadcastable(__VA_ARGS__)
#else
#define neml_assert_broadcastable_dbg(...) ((void)0)
#endif
///@}

/// Assert that all tensors are dynamic-broadcastable
///@{
template <class... T>
void neml_assert_dynamic_broadcastable(const T &...);

#ifndef NDEBUG
#define neml_assert_dynamic_broadcastable_dbg(...)                                                 \
  ::neml2::neml_assert_dynamic_broadcastable(__VA_ARGS__)
#else
#define neml_assert_dynamic_broadcastable_dbg(...) ((void)0)
#endif
///@}

/// Assert that all tensors are intermediate-broadcastable
///@{
template <class... T>
void neml_assert_intmd_broadcastable(const T &...);

#ifndef NDEBUG
#define neml_assert_intmd_broadcastable_dbg(...) ::neml2::neml_assert_broadcastable(__VA_ARGS__)
#else
#define neml_assert_intmd_broadcastable_dbg(...) ((void)0)
#endif
///@}

/// Assert that all tensors are batch-broadcastable
///@{
template <class... T>
void neml_assert_batch_broadcastable(const T &...);

#ifndef NDEBUG
#define neml_assert_batch_broadcastable_dbg(...)                                                   \
  ::neml2::neml_assert_batch_broadcastable(__VA_ARGS__)
#else
#define neml_assert_batch_broadcastable_dbg(...) ((void)0)
#endif
///@}

/// Assert that all tensors are base-broadcastable
///@{
template <class... T>
void neml_assert_base_broadcastable(const T &...);

#ifndef NDEBUG
#define neml_assert_base_broadcastable_dbg(...) ::neml2::neml_assert_base_broadcastable(__VA_ARGS__)
#else
#define neml_assert_base_broadcastable_dbg(...) ((void)0)
#endif
///@}

/// Assert that all tensors are static-broadcastable
///@{
template <class... T>
void neml_assert_static_broadcastable(const T &...);

#ifndef NDEBUG
#define neml_assert_static_broadcastable_dbg(...)                                                  \
  ::neml2::neml_assert_static_broadcastable(__VA_ARGS__)
#else
#define neml_assert_static_broadcastable_dbg(...) ((void)0)
#endif
///@}
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
  neml_assert(utils::broadcastable(tensors...),
              "The ",
              sizeof...(tensors),
              " operands are not broadcastable. The dynamic shapes are ",
              tensors.dynamic_sizes()...,
              ", the intermediate shapes are ",
              tensors.intmd_sizes()...,
              ", and the base shapes are ",
              tensors.base_sizes()...);
}

template <class... T>
void
neml_assert_dynamic_broadcastable(const T &... tensors)
{
  neml_assert(utils::dynamic_broadcastable(tensors...),
              "The ",
              sizeof...(tensors),
              " operands are not dynamic-broadcastable. The dynamic shapes are ",
              tensors.dynamic_sizes()...);
}

template <class... T>
void
neml_assert_intmd_broadcastable(const T &... tensors)
{
  neml_assert(utils::intmd_broadcastable(tensors...),
              "The ",
              sizeof...(tensors),
              " operands are not intermediate-broadcastable. The intermediate shapes are ",
              tensors.intmd_sizes()...);
}

template <class... T>
void
neml_assert_batch_broadcastable(const T &... tensors)
{
  neml_assert(utils::dynamic_broadcastable(tensors...) && utils::intmd_broadcastable(tensors...),
              "The ",
              sizeof...(tensors),
              " operands are not batch-broadcastable. The dynamic shapes are ",
              tensors.dynamic_sizes()...,
              ", and the intermediate shapes are ",
              tensors.intmd_sizes()...);
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
neml_assert_static_broadcastable(const T &... tensors)
{
  neml_assert(utils::intmd_broadcastable(tensors...) && utils::base_broadcastable(tensors...),
              "The ",
              sizeof...(tensors),
              " operands are not static-broadcastable. The intermediate shapes are ",
              tensors.intmd_sizes()...,
              ", and the base shapes are ",
              tensors.base_sizes()...);
}
} // namespace neml2
