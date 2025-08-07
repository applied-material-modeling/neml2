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

#include <catch2/matchers/catch_matchers_templated.hpp>

#include "neml2/tensors/indexing.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/from_assembly.h"

namespace test
{
template <typename T>
class TensorMatcher : public Catch::Matchers::MatcherBase<T>
{
public:
  explicit TensorMatcher(T expected, double rtol, double atol);
  bool match(const T & actual) const override;
  std::string describe() const override;

private:
  const T _m_expected;
  mutable T _m;

  mutable bool _shapes_match = false;
  mutable bool _devices_match = false;
  mutable bool _dtypes_match = false;

  const double _rtol;
  const double _atol;
  mutable bool _allclose = false;
};

template <typename T>
TensorMatcher<T>
allclose(const T & expected, double rtol = 1e-5, std::optional<double> atol = std::nullopt);

namespace details
{
/// Check if two tensors are equal. In case of floating point tensors, use allclose (with tolerances)
bool allclose(const neml2::ATensor & a, const neml2::ATensor & b, double rtol, double atol);
} // namespace details
} // namespace test

/// Generic main function for the test suite of name \p name.
int test_main(int argc, char * argv[], const std::string & name);

/**
 * @brief A utility function to guess the path to the test directory based on a hint and some
 * heuristics.
 *
 * Given the stem of the test directory \p stem, the \p hint is considered valid if it exists and
 * ends with \p stem.
 *
 * If the hint is not valid, we traverse up to 4 levels from the executable path as
 * alternative hints, i.e.,
 * 1. hint = {exec_prefix}/../../../..
 * 2. hint = {exec_prefix}/../../..
 * 3. hint = {exec_prefix}/../..
 * 4. hint = {exec_prefix}/..
 * 5. hint = {exec_prefix}
 * where {exec_prefix} is the directory where the executable is located.
 *
 * A nonzero return code is returned if the test directory is not found.
 *
 * Upon exit, the \p hint is updated to the absolute path of the test directory if error code is 0.
 *
 * @param stem The stem of the test directory, e.g., unit, regression, etc.
 * @param hint The hint to the test directory
 */
int
guess_test_dir(const std::string & stem, std::string & hint, const std::string & exec_prefix = "");

/// Get test suite additional devices
const std::unordered_set<neml2::Device> & get_test_suite_additional_devices();

/**
 * @brief Parse the cliarg (a comma-separated list of device specs) into a set of devices
 *
 * @return int Error code:
 * 0: success
 * 1: invalid device spec
 * 2: invalid device type (e.g., contains CPU)
 * 3: duplicate device spec
 */
int init_test_devices(const std::string & additional_devs);

/**
 * @brief A simple finite-differencing helper to numerically approximate the derivative of the
 * function at the given point.
 *
 * @tparam F The functor to differentiate
 * @tparam T Type of the input variable, must be _batched_
 * @param f The functor to differentiate, must accept the input of type `Tensor`
 * @param x The point where the derivative is evaluated
 * @param eps The relative perturbation (for each component in the case of non-Scalar input)
 * @param aeps The minimum perturbation to improve numerical stability
 * @return Tensor The derivative at the given point approximated using finite differencing
 */
template <typename F>
[[nodiscard]] neml2::Tensor
finite_differencing_derivative(F && f,
                               const neml2::Tensor & x,
                               double eps = 1e-6,
                               std::optional<double> aeps = std::nullopt)
{
  using namespace neml2;

  // Set absolute perturbation based on dtype
  double aeps_default = std::sqrt(neml2::machine_precision(x.scalar_type()));
  double aeps2 = aeps.value_or(aeps_default);

  // Flatten x to support arbitrarily shaped input
  auto xf = x.static_flatten();
  auto y0 = Tensor(std::forward<F>(f)(x)).clone();
  auto y_intmd_sizes = y0.intmd_sizes().vec();
  auto y_base_sizes = y0.base_sizes().vec();
  y0 = y0.static_flatten();
  auto dyf_dxf = std::vector<Tensor>(xf.base_size(0));
  for (Size i = 0; i < xf.base_size(0); i++)
  {
    auto dx = eps * Scalar(abs(xf.base_index({i})));
    dx.index_put_({dx < aeps2}, aeps2);

    auto xf1 = xf.clone();
    xf1.base_index_put_({i}, xf1.base_index({i}) + dx);
    auto x1 = xf1.static_reshape(x.intmd_sizes(), x.base_sizes());

    auto y1 = Tensor(std::forward<F>(f)(x1)).clone().static_flatten();
    dyf_dxf[i] = (y1 - y0) / dx;
  }

  // Reshape the derivative back to the correct shape
  auto dy_dx = base_stack(dyf_dxf, -1);
  return from_assembly<2>(dy_dx, {y_intmd_sizes, x.intmd_sizes()}, {y_base_sizes, x.base_sizes()});
}
