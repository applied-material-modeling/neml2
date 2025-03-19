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

#include <array>

#include "neml2/misc/types.h"
#include "neml2/misc/defaults.h"

namespace neml2
{
class Tensor;

constexpr std::array<std::array<Size, 3>, 3> mandel_reverse_index{
    std::array<Size, 3>{0, 5, 4}, std::array<Size, 3>{5, 1, 3}, std::array<Size, 3>{4, 3, 2}};
constexpr std::array<std::array<Size, 2>, 6> mandel_index{std::array<Size, 2>{0, 0},
                                                          std::array<Size, 2>{1, 1},
                                                          std::array<Size, 2>{2, 2},
                                                          std::array<Size, 2>{1, 2},
                                                          std::array<Size, 2>{0, 2},
                                                          std::array<Size, 2>{0, 1}};

constexpr std::array<std::array<Size, 3>, 3> skew_reverse_index{
    std::array<Size, 3>{0, 2, 1}, std::array<Size, 3>{2, 0, 0}, std::array<Size, 3>{1, 0, 0}};
constexpr std::array<std::array<Real, 3>, 3> skew_factor{std::array<Real, 3>{0.0, -1.0, 1.0},
                                                         std::array<Real, 3>{1.0, 0.0, -1.0},
                                                         std::array<Real, 3>{-1.0, 1.0, 0.0}};

inline constexpr Real
mandel_factor(Size i)
{
  return i < 3 ? 1.0 : sqrt2;
}

const Tensor & full_to_mandel_map(const TensorOptions & opt = default_integer_tensor_options());
const Tensor & mandel_to_full_map(const TensorOptions & opt = default_integer_tensor_options());
const Tensor & full_to_mandel_factor(const TensorOptions & opt = default_tensor_options());
const Tensor & mandel_to_full_factor(const TensorOptions & opt = default_tensor_options());
const Tensor & full_to_skew_map(const TensorOptions & opt = default_integer_tensor_options());
const Tensor & skew_to_full_map(const TensorOptions & opt = default_integer_tensor_options());
const Tensor & full_to_skew_factor(const TensorOptions & opt = default_tensor_options());
const Tensor & skew_to_full_factor(const TensorOptions & opt = default_tensor_options());

/**
 * @brief Generic function to reduce two axes to one with some map
 *
 * The tensor in full notation \p full can have arbitrary batch shape. The optional argument \p dim
 * denotes the base dimension starting from which the conversion should take place.
 *
 * The function will reduce the two axis at the desired location down to one, using the provided
 * maps.
 *
 * For example, a full tensor has shape `(2, 3, 1, 5; 2, 9, 3, 3, 2, 3)` where the semicolon
 * separates batch and base shapes. The *reduction* axes have base dim 2 and 3. After applying the
 * reduction, the resulting tensor will have shape `(2, 3, 1, 5; 2, 9, X, 2, 3)` where X
 * is the reduced shape. In this example, the base dim (the
 * second argument to this function) should be 2.
 *
 * @param full The input tensor in full notation
 * @param rmap The reduction map
 * @param rfactors The reduction factors
 * @param dim The base dimension where the reduced axes start
 * @return Tensor The reduced tensor
 */
Tensor
full_to_reduced(const Tensor & full, const Tensor & rmap, const Tensor & rfactors, Size dim = 0);

/**
 * @brief Convert a Tensor from reduced notation to full notation.
 *
 * See @ref full_to_reduced for a detailed explanation.
 *
 * @param reduced The input tensor in reduced notation
 * @param rmap The unreduction map
 * @param rfactors The unreduction factors
 * @param dim The base dimension where the reduced axes start
 * @return Tensor The resulting tensor in full notation.
 */
Tensor
reduced_to_full(const Tensor & reduced, const Tensor & rmap, const Tensor & rfactors, Size dim = 0);

/**
 * @brief Convert a `Tensor` from full notation to Mandel notation.
 *
 * The tensor in full notation \p full can have arbitrary batch shape. The optional argument \p dim
 * denotes the base dimension starting from which the conversion should take place.
 *
 * For example, a full tensor has shape `(2, 3, 1, 5; 2, 9, 3, 3, 2, 3)` where the semicolon
 * separates batch and base shapes. The *symmetric* axes have base dim 2 and 3. After converting to
 * Mandel notation, the resulting tensor will have shape `(2, 3, 1, 5; 2, 9, 6, 2, 3)`. Note how the
 * shape of the symmetric dimensions `(3, 3)` becomes `(6)`. In this example, the base dim (the
 * second argument to this function) should be 2.
 *
 * @param full The input tensor in full notation
 * @param dim The base dimension where the symmetric axes start
 * @return Tensor The resulting tensor using Mandel notation to represent the symmetric axes.
 */
Tensor full_to_mandel(const Tensor & full, Size dim = 0);

/**
 * @brief Convert a Tensor from Mandel notation to full notation.
 *
 * See @ref full_to_mandel for a detailed explanation.
 *
 * @param mandel The input tensor in Mandel notation
 * @param dim The base dimension where the symmetric axes start
 * @return Tensor The resulting tensor in full notation.
 */
Tensor mandel_to_full(const Tensor & mandel, Size dim = 0);

/**
 * @brief Convert a `Tensor` from full notation to skew vector notation.
 *
 * The tensor in full notation \p full can have arbitrary batch shape. The optional argument \p dim
 * denotes the base dimension starting from which the conversion should take place.
 *
 * For example, a full tensor has shape `(2, 3, 1, 5; 2, 9, 3, 3, 2, 3)` where the semicolon
 * separates batch and base shapes. The *symmetric* axes have base dim 2 and 3. After converting to
 * skew notation, the resulting tensor will have shape `(2, 3, 1, 5; 2, 9, 3, 2, 3)`. Note how the
 * shape of the symmetric dimensions `(3, 3)` becomes `(3)`. In this example, the base dim (the
 * second argument to this function) should be 2.
 *
 * @param full The input tensor in full notation
 * @param dim The base dimension where the symmetric axes start
 * @return Tensor The resulting tensor using skew notation to represent the skew-symmetric
 * axes.
 */
Tensor full_to_skew(const Tensor & full, Size dim = 0);

/**
 * @brief Convert a Tensor from skew vector notation to full notation.
 *
 * See @ref full_to_skew for a detailed explanation.
 *
 * @param skew The input tensor in skew notation
 * @param dim The base dimension where the symmetric axes start
 * @return Tensor The resulting tensor in full notation.
 */
Tensor skew_to_full(const Tensor & skew, Size dim = 0);
} // namespace neml2
