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

#include "neml2/models/solid_mechanics/elasticity/IsotropicElasticityConverter.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
const IsotropicElasticityConverter::ConversionTableType IsotropicElasticityConverter::table = {
    {{ElasticConstant::BULK_MODULUS, ElasticConstant::SHEAR_MODULUS},
     {&IsotropicElasticityConverter::K_G_to_K, &IsotropicElasticityConverter::K_G_to_G}},
    {{ElasticConstant::YOUNGS_MODULUS, ElasticConstant::POISSONS_RATIO},
     {&IsotropicElasticityConverter::E_nu_to_K, &IsotropicElasticityConverter::E_nu_to_G}}};

IsotropicElasticityConverter::ConversionType
IsotropicElasticityConverter::K_G_to_K(const InputType & input, const DerivativeFlagType & deriv)
{
  const auto & K = input[0];

  const auto dK_dK = deriv[0] ? imap<Scalar>(K.options()).as<Scalar>() : Scalar();
  const auto dK_dG = deriv[1] ? Scalar::zeros(K.options()) : Scalar();

  return {K, {dK_dK, dK_dG}};
}

IsotropicElasticityConverter::ConversionType
IsotropicElasticityConverter::K_G_to_G(const InputType & input, const DerivativeFlagType & deriv)
{
  const auto & G = input[1];

  const auto dG_dK = deriv[0] ? Scalar::zeros(G.options()) : Scalar();
  const auto dG_dG = deriv[1] ? imap<Scalar>(G.options()).as<Scalar>() : Scalar();

  return {G, {dG_dK, dG_dG}};
}

IsotropicElasticityConverter::ConversionType
IsotropicElasticityConverter::E_nu_to_K(const InputType & input, const DerivativeFlagType & deriv)
{
  const auto & E = input[0];
  const auto & nu = input[1];

  const auto K = E / 3 / (1 - 2 * nu);
  const auto dK_dE = deriv[0] ? K / E : Scalar();
  const auto dK_dnu = deriv[1] ? 6 * K * K / E : Scalar();

  return {K, {dK_dE, dK_dnu}};
}

IsotropicElasticityConverter::ConversionType
IsotropicElasticityConverter::E_nu_to_G(const InputType & input, const DerivativeFlagType & deriv)
{
  const auto & E = input[0];
  const auto & nu = input[1];

  const auto G = E / (2 * (1 + nu));
  const auto dG_dE = deriv[0] ? G / E : Scalar();
  const auto dG_dnu = deriv[1] ? -G / (1 + nu) : Scalar();

  return {G, {dG_dE, dG_dnu}};
}

} // namespace neml2
