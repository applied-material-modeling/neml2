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
#include "neml2/base/MultiEnumSelection.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
/**
 * @brief Interface for objects defining elasticity tensors in terms of other parameters
 *
 * @tparam N Number of independent elastic constants
 */
template <class Derived, std::size_t N>
class ElasticityInterface : public Derived
{
public:
  static OptionSet expected_options();

  ElasticityInterface(const OptionSet & options);

protected:
  /// Declare elastic constants (by resolving cross-references)
  void declare_elastic_constant(ElasticConstant);

  /// Input elastic constant types (ordered according to ElasticConstant)
  std::array<ElasticConstant, N> _constant_types;

  /// Input elastic constants (ordered according to ElasticConstant)
  std::array<const Scalar *, N> _constants;

  /// Whether we need to calculate the derivative of the constants
  std::array<bool, N> _need_derivs;

private:
  /// Input coefficients (without reordering)
  const std::vector<TensorName> _coefs;

  /// Input coefficient types (without reordering)
  const std::vector<ElasticConstant> _coef_types;

  /// Flags to indicate whether coefficients are parameters or buffers
  std::vector<bool> _coef_as_param;

  /// Helper counter to fill out std::array
  std::size_t _counter = 0;
};

} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
template <class Derived, std::size_t N>
OptionSet
ElasticityInterface<Derived, N>::expected_options()
{
  OptionSet options = Derived::expected_options();

  MultiEnumSelection type_selection({"LAME_LAMBDA",
                                     "BULK_MODULUS",
                                     "SHEAR_MODULUS",
                                     "YOUNGS_MODULUS",
                                     "POISSONS_RATIO",
                                     "P_WAVE_MODULUS",
                                     "INVALID"},
                                    {static_cast<int>(ElasticConstant::LAME_LAMBDA),
                                     static_cast<int>(ElasticConstant::BULK_MODULUS),
                                     static_cast<int>(ElasticConstant::SHEAR_MODULUS),
                                     static_cast<int>(ElasticConstant::YOUNGS_MODULUS),
                                     static_cast<int>(ElasticConstant::POISSONS_RATIO),
                                     static_cast<int>(ElasticConstant::P_WAVE_MODULUS),
                                     static_cast<int>(ElasticConstant::INVALID)},
                                    {"INVALID"});
  options.set<MultiEnumSelection>("coefficient_types") = type_selection;
  options.set("coefficient_types").doc() =
      "Types for each parameter, options are: " + type_selection.candidates_str();

  options.set_parameter<std::vector<TensorName>>("coefficients");
  options.set("coefficients").doc() = "Coefficients used to define the elasticity tensor";

  options.set<std::vector<bool>>("coefficient_as_parameter") = {true};
  options.set("coefficient_as_parameter").doc() =
      "Whether to treat the coefficients as (trainable) parameters. Default is true. Setting this "
      "option to false will treat the coefficients as buffers.";

  return options;
}

template <class Derived, std::size_t N>
ElasticityInterface<Derived, N>::ElasticityInterface(const OptionSet & options)
  : Derived(options),
    _coefs(options.get<std::vector<TensorName>>("coefficients")),
    _coef_types(options.get<MultiEnumSelection>("coefficient_types").as<ElasticConstant>()),
    _coef_as_param(options.get<std::vector<bool>>("coefficient_as_parameter"))
{
  if (_coefs.size() != N)
    throw NEMLException("Expected " + std::to_string(N) + " coefficients, got " +
                        std::to_string(_coefs.size()) + ".");
  if (_coef_types.size() != N)
    throw NEMLException("Expected " + std::to_string(N) + " coefficient types, got " +
                        std::to_string(_coef_types.size()) + ".");
  if (_coef_as_param.size() != 1 && _coef_as_param.size() != N)
    throw NEMLException("Expected 1 or " + std::to_string(N) +
                        " entries in coefficient_as_parameter, got " +
                        std::to_string(_coef_as_param.size()) + ".");

  if (_coef_as_param.size() == 1)
    _coef_as_param.resize(N, _coef_as_param[0]);

  // Fill out _constants, _constant_types, and _constant_names by sorting the coefficients according
  // to the order defined by ElasticConstant
  declare_elastic_constant(ElasticConstant::LAME_LAMBDA);
  declare_elastic_constant(ElasticConstant::BULK_MODULUS);
  declare_elastic_constant(ElasticConstant::SHEAR_MODULUS);
  declare_elastic_constant(ElasticConstant::YOUNGS_MODULUS);
  declare_elastic_constant(ElasticConstant::POISSONS_RATIO);
  declare_elastic_constant(ElasticConstant::P_WAVE_MODULUS);

  // Figure out which coefficients need derivatives
  for (std::size_t i = 0; i < _constant_types.size(); i++)
    _need_derivs[i] = (Derived::nl_param(neml2::name(_constant_types[i])) != nullptr);
}

template <class Derived, std::size_t N>
void
ElasticityInterface<Derived, N>::declare_elastic_constant(ElasticConstant ptype)
{
  for (std::size_t i = 0; i < _coefs.size(); i++)
  {
    if (_coef_types[i] == ElasticConstant::INVALID)
      throw NEMLException("Invalid coefficient type provided for coefficient " + std::to_string(i) +
                          ".");

    if (_coef_types[i] != ptype)
      continue;

    if (std::find(_constant_types.begin(), std::next(_constant_types.begin(), _counter), ptype) !=
        std::next(_constant_types.begin(), _counter))
      throw NEMLException("Duplicate coefficient type provided for coefficient " +
                          std::to_string(i) + ".");

    const auto pname = neml2::name(ptype);
    const auto * pval = _coef_as_param[i]
                            ? &Derived::template declare_parameter<Scalar>(
                                  pname, _coefs[i], /*allow_nonlinear*/ true)
                            : &Derived::template declare_buffer<Scalar>(pname, _coefs[i]);

    if (_counter >= N)
      throw NEMLException("Too many coefficients provided.");
    _constant_types[_counter] = ptype;
    _constants[_counter] = pval;
    _counter++;

    return;
  }
}
} // namespace neml2
