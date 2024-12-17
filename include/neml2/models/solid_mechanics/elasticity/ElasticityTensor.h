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

#include "neml2/models/NonlinearParameter.h"
#include "neml2/models/solid_mechanics/elasticity/ElasticityConverter.h"

namespace neml2
{
/**
 * @brief Superclass for defining elasticity tensors in terms of other parameters
 *
 * @tparam N Number of independent elastic constants
 */
template <std::size_t N>
class ElasticityTensor : public NonlinearParameter<SSR4>
{
public:
  static OptionSet expected_options();

  ElasticityTensor(const OptionSet & options);

protected:
  /// Declare elastic constants (by resolving cross-references)
  void declare_elastic_constant(LameParameter);

  /// Input elastic constant types (ordered according to LameParameter)
  std::array<LameParameter, N> _constant_types;

  /// Input elastic constants (ordered according to LameParameter)
  std::array<const Scalar *, N> _constants;

  /// Whether we need to calculate the derivative of the constants
  std::array<bool, N> _need_derivs;

private:
  /// Input coefficients (without reordering)
  const std::vector<CrossRef<Scalar>> _coefs;

  /// Input coefficient types (without reordering)
  const std::vector<LameParameter> _coef_types;

  /// Flags to indicate whether coefficients are parameters or buffers
  std::vector<bool> _coef_as_param;

  /// Helper counter to fill out std::array
  std::size_t _counter = 0;
};

} // namespace neml2
