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

#include "neml2/tensors/Scalar.h"

#include <tuple>
#include <map>
#include <unordered_map>
#include <numeric>
#include <algorithm>

namespace neml2
{
enum class ElasticConstant : std::uint8_t
{
  INVALID = 0,
  // Isotropic
  LAME_LAMBDA = 1,
  BULK_MODULUS = 2,
  SHEAR_MODULUS = 3,
  YOUNGS_MODULUS = 4,
  POISSONS_RATIO = 5,
  P_WAVE_MODULUS = 6,
  // Cubic symmetry
  CUBIC_C1 = 7,
  CUBIC_C2 = 8,
  CUBIC_C3 = 9
};

std::string name(ElasticConstant p);

/**
 * @brief Base class for converters responsible for converting between different parameterizations
 * of the linear elasticity tensor in different symmetry groups
 *
 * @tparam N Number of independent elastic constants
 */
template <std::size_t N>
class ElasticityConverter
{
public:
  using InputType = std::array<Scalar, N>;
  using InputPtrType = std::array<const Scalar *, N>;
  using DerivativeFlagType = std::array<bool, N>;
  using DerivativeType = std::array<Scalar, N>;
  using ConversionType = std::pair<Scalar, DerivativeType>;
  using ResultType = std::array<ConversionType, N>;

  using ConverterKey = std::array<ElasticConstant, N>;
  using ConverterType = ConversionType (*)(const InputType &, const DerivativeFlagType &);
  using ConversionTableType = std::map<ConverterKey, std::array<ConverterType, N>>;

  ElasticityConverter(const ConversionTableType & table,
                      const ConverterKey & output_parameterization,
                      const ConverterKey & input_parameterization,
                      const DerivativeFlagType & deriv_requested)
    : _output_parameterization(output_parameterization),
      _deriv_requested(deriv_requested)
  {
    assert_ascending(output_parameterization);
    assert_ascending(input_parameterization);
    if (!table.count(input_parameterization))
      throw NEMLException(
          "Parameterization not found in the conversion table. This typically means that the "
          "given combination of Lame parameters is not yet supported.");
    _converters = table.at(input_parameterization);
  }

  /// Convert input to independent elastic constants with derivatives
  ResultType convert(const InputType & input) const
  {
    ResultType ret{};
    for (std::size_t i = 0; i < N; ++i)
      ret[i] = _converters[i](input, _deriv_requested);
    return ret;
  }

  /// Convert input to independent elastic constants with derivatives
  ResultType convert(const InputPtrType & input) const
  {
    InputType input_values{};
    for (std::size_t i = 0; i < N; ++i)
      input_values[i] = *input[i];
    return convert(input_values);
  }

  /// Convert input to a single elastic constant with derivatives
  ConversionType convert(const InputType & input, const ElasticConstant p) const
  {
    return _converters[find_index(p)](input, _deriv_requested);
  }

  /// Convert input to a single elastic constant with derivatives
  ConversionType convert(const InputPtrType & input, const ElasticConstant p) const
  {
    InputType input_values{};
    for (std::size_t i = 0; i < N; ++i)
      input_values[i] = *input[i];
    return convert(input_values, p);
  }

private:
  /// Generate ordering of the input parameters
  void assert_ascending(const ConverterKey & ps) const
  {
    for (std::size_t i = 1; i < N; ++i)
      if (static_cast<std::uint8_t>(ps[i]) <= static_cast<std::uint8_t>(ps[i - 1]))
        throw NEMLException(
            "Internal error: ElasticityConverters only accept Lame parameters sorted in the "
            "following order: LAME_LAMBDA, BULK_MODULUS, SHEAR_MODULUS, YOUNGS_MODULUS, "
            "POISSONS_RATIO, P_WAVE_MODULUS.");
  }

  /// Find the index of a Lame parameter in the output parameterization
  std::size_t find_index(ElasticConstant p) const
  {
    auto it = std::find(_output_parameterization.begin(), _output_parameterization.end(), p);
    if (it == _output_parameterization.end())
      throw NEMLException(
          "Internal error: Lame parameter not found in the output parameterization.");
    return std::distance(_output_parameterization.begin(), it);
  }

  /// Output parameterization
  const ConverterKey _output_parameterization;

  /// Flags to indicate which derivatives are requested
  const DerivativeFlagType _deriv_requested = {};

  /// Converter
  std::array<ConverterType, N> _converters;
};

} // namespace neml2
