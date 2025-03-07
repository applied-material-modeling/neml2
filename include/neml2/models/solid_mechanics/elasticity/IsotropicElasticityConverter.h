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

#include "neml2/models/solid_mechanics/elasticity/ElasticityConverter.h"

namespace neml2
{
/**
 * @brief Converter for linearized elastic constants assuming isotropic symmetry
 */
class IsotropicElasticityConverter : public ElasticityConverter<2>
{
public:
  ///@{
  /// @name Conversion functions from various parameterizations to K and mu
  static ConversionType K_G_to_K(const InputType &, const DerivativeFlagType &);
  static ConversionType K_G_to_G(const InputType &, const DerivativeFlagType &);
  static ConversionType E_nu_to_K(const InputType &, const DerivativeFlagType &);
  static ConversionType E_nu_to_G(const InputType &, const DerivativeFlagType &);
  ///@}

  /// Conversion table
  static const ConversionTableType table;

  IsotropicElasticityConverter(const ConverterKey & parameterization,
                               const DerivativeFlagType & deriv_requested)
    : ElasticityConverter<2>(table,
                             {ElasticConstant::BULK_MODULUS, ElasticConstant::SHEAR_MODULUS},
                             parameterization,
                             deriv_requested)
  {
  }
};

} // namespace neml2
