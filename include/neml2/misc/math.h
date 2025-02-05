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
class SR2;
class WR2;
class SWR4;
class SSR4;
class WSR4;
namespace math
{
constexpr Real eps = std::numeric_limits<at::scalar_value_type<Real>::type>::epsilon();

constexpr Real sqrt2 = 1.4142135623730951;
constexpr Real invsqrt2 = 0.7071067811865475;

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

/**
 * @brief A helper class to hold static data of type torch::Tensor
 *
 * This class exists because torch::Tensor cannot be declared as constexpr nor as static data in the
 * global scope. The former is obvious. The latter is because at the time static variables are
 * initialized, some torch data structures have not been properly initialized yet.
 *
 */
struct ConstantTensors
{
  ConstantTensors();

  // Get the global constants
  static ConstantTensors & get();

  static const torch::Tensor & full_to_mandel_map();
  static const torch::Tensor & mandel_to_full_map();
  static const torch::Tensor & full_to_mandel_factor();
  static const torch::Tensor & mandel_to_full_factor();
  static const torch::Tensor & full_to_skew_map();
  static const torch::Tensor & skew_to_full_map();
  static const torch::Tensor & full_to_skew_factor();
  static const torch::Tensor & skew_to_full_factor();

private:
  torch::Tensor _full_to_mandel_map;
  torch::Tensor _mandel_to_full_map;
  torch::Tensor _full_to_mandel_factor;
  torch::Tensor _mandel_to_full_factor;
  torch::Tensor _full_to_skew_map;
  torch::Tensor _skew_to_full_map;
  torch::Tensor _full_to_skew_factor;
  torch::Tensor _skew_to_full_factor;
};

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
Tensor full_to_reduced(const Tensor & full,
                       const torch::Tensor & rmap,
                       const torch::Tensor & rfactors,
                       Size dim = 0);

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
Tensor reduced_to_full(const Tensor & reduced,
                       const torch::Tensor & rmap,
                       const torch::Tensor & rfactors,
                       Size dim = 0);

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

/**
 * @brief Use automatic differentiation (AD) to calculate the derivatives of a Tensor w.r.t. another
 * Tensor
 *
 * @warning Torch (and hence NEML2) AD wasn't designed to compute the full Jacobian from the very
 * beginning. Using this method to calculate the full Jacobian is inefficient and is subjected to
 * some restrictions on batch shapes.
 *
 * @param y The `Tensor` to to be differentiated
 * @param xs The arguments to take derivatives with respect to
 * @param retain_graph Whether to retain the computation graph (necessary if y has base storage size
 * > 1)
 * @param create_graph Whether to create the computation graph (necessary if you want to
 * differentiate the returned Jacobian)
 * @param allow_unused Whether to allow unused input argument \p x
 * @return Tensor \f$\partial y/\partial p\f$
 */
std::vector<Tensor> jacrev(const Tensor & y,
                           const std::vector<Tensor> & xs,
                           bool retain_graph = false,
                           bool create_graph = false,
                           bool allow_unused = false);

/// Similar to the other jacrev, but for a single input
Tensor jacrev(const Tensor & y,
              const Tensor & x,
              bool retain_graph = false,
              bool create_graph = false,
              bool allow_unused = false);

Tensor base_diag_embed(const Tensor & a, Size offset = 0, Size d1 = -2, Size d2 = -1);

/// Product w_ik e_kj - e_ik w_kj with e SR2 and w WR2
SR2 skew_and_sym_to_sym(const SR2 & e, const WR2 & w);

/// Derivative of w_ik e_kj - e_ik w_kj wrt. e
SSR4 d_skew_and_sym_to_sym_d_sym(const WR2 & w);

/// Derivative of w_ik e_kj - e_ik w_kj wrt. w
SWR4 d_skew_and_sym_to_sym_d_skew(const SR2 & e);

/// Shortcut product a_ik b_kj - b_ik a_kj with both SR2
WR2 multiply_and_make_skew(const SR2 & a, const SR2 & b);

/// Derivative of a_ik b_kj - b_ik a_kj wrt a
WSR4 d_multiply_and_make_skew_d_first(const SR2 & b);

/// Derivative of a_ik b_kj - b_ik a_kj wrt b
WSR4 d_multiply_and_make_skew_d_second(const SR2 & a);

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
batch_cat(const std::vector<T> & tensors, Size d = 0)
{
  neml_assert_dbg(!tensors.empty(), "batch_cat must be given at least one tensor");
  std::vector<torch::Tensor> torch_tensors(tensors.begin(), tensors.end());
  auto d2 = d >= 0 ? d : d - tensors.begin()->base_dim();
  return T(torch::cat(torch_tensors, d2), tensors.begin()->batch_dim());
}

neml2::Tensor base_cat(const std::vector<Tensor> & tensors, Size d = 0);

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
batch_stack(const std::vector<T> & tensors, Size d = 0)
{
  neml_assert_dbg(!tensors.empty(), "batch_stack must be given at least one tensor");
  std::vector<torch::Tensor> torch_tensors(tensors.begin(), tensors.end());
  auto d2 = d >= 0 ? d : d - tensors.begin()->base_dim();
  return T(torch::stack(torch_tensors, d2), tensors.begin()->batch_dim() + 1);
}

neml2::Tensor base_stack(const std::vector<Tensor> & tensors, Size d = 0);

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
batch_sum(const T & a, Size d = 0)
{
  neml_assert_dbg(a.batch_dim() > 0, "Must have a batch dimension to sum along");
  auto d2 = d >= 0 ? d : d - a.base_dim();
  return T(torch::sum(a, d2), a.batch_sizes().slice(0, -1));
}

neml2::Tensor base_sum(const Tensor & a, Size d);

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
batch_mean(const T & a, Size d = 0)
{
  neml_assert_dbg(a.batch_dim() > 0, "Must have a batch dimension to take average");
  auto d2 = d >= 0 ? d : d - a.base_dim();
  return T(torch::mean(a, d2), a.batch_sizes().slice(0, -1));
}

neml2::Tensor base_mean(const Tensor & a, Size d);

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
pow(const T & a, const Real & n)
{
  return T(torch::pow(a, n), a.batch_sizes());
}

Tensor pow(const Real & a, const Tensor & n);

Tensor pow(const Tensor & a, const Tensor & n);

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
sign(const T & a)
{
  return T(torch::sign(a), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
cosh(const T & a)
{
  return T(torch::cosh(a), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
sinh(const T & a)
{
  return T(torch::sinh(a), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
tanh(const T & a)
{
  return T(torch::tanh(a), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
arccos(const T & a)
{
  return T(torch::arccos(a), a.batch_dim());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
arcsin(const T & a)
{
  return T(torch::arcsin(a), a.batch_dim());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
where(const torch::Tensor & condition, const T & a, const T & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return T(torch::where(condition, a, b), broadcast_batch_dim(a, b));
}

/**
 * This is (almost) equivalent to Torch's heaviside, except that the Torch's version is not
 * differentiable (back-propagatable). I said "almost" because torch::heaviside allows you to set
 * the return value in the case of input == 0. Our implementation always return 0.5 when the input
 * == 0.
 */
template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
heaviside(const T & a)
{
  return (sign(a) + 1.0) / 2.0;
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
macaulay(const T & a)
{
  return T(torch::Tensor(a) * torch::Tensor(heaviside(a)), a.batch_dim());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
dmacaulay(const T & a)
{
  return heaviside(a);
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
sqrt(const T & a)
{
  return T(torch::sqrt(a), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
exp(const T & a)
{
  return T(torch::exp(a), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
abs(const T & a)
{
  return T(torch::abs(a), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
diff(const T & a, Size n = 1, Size dim = -1)
{
  return T(torch::diff(a, n, dim), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
batch_diag_embed(const T & a, Size offset = 0, Size d1 = -2, Size d2 = -1)
{
  return T(torch::diag_embed(
               a, offset, d1 < 0 ? d1 - a.base_dim() : d1, d2 < 0 ? d2 - a.base_dim() : d2),
           a.batch_dim() + 1);
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
log(const T & a)
{
  return T(torch::log(a), a.batch_sizes());
}

template <class T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
T
log10(const T & a)
{
  return T(torch::log10(a), a.batch_sizes());
}

namespace linalg
{
/// Vector norm of a vector. Falls back to math::abs is \p v is a Scalar.
Tensor vector_norm(const Tensor & v);

/// Inverse of a square matrix
Tensor inv(const Tensor & m);

/// Solve the linear system A X = B
Tensor solve(const Tensor & A, const Tensor & B);

std::tuple<Tensor, Tensor> lu_factor(const Tensor & A, bool pivot = true);

Tensor lu_solve(const Tensor & LU,
                const Tensor & pivots,
                const Tensor & B,
                bool left = true,
                bool adjoint = false);
} // namespace linalg
} // namespace math
} // namespace neml2
