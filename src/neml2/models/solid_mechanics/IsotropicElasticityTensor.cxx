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

#include "neml2/models/solid_mechanics/IsotropicElasticityTensor.h"

namespace neml2
{
register_NEML2_object(IsotropicElasticityTensor);

OptionSet
IsotropicElasticityTensor::expected_options()
{
  OptionSet options = ElasticityTensor::expected_options();
  options.doc() += "  This class defines an isotropic elasticity tensor using two parameters."
                   "  Various options are available for which two parameters to provide.";

  options.set_parameter<CrossRef<Scalar>>("p1");
  options.set("p1").doc() = "First parameter";

  EnumSelection type_selection({"youngs_modulus", "poissons_ratio", "INVALID"},
                               {static_cast<int>(IsotropicElasticityTensor::ParamType::YOUNGS),
                                static_cast<int>(IsotropicElasticityTensor::ParamType::POISSONS),
                                static_cast<int>(IsotropicElasticityTensor::ParamType::INVALID)},
                               "INVALID");
  options.set<EnumSelection>("p1_type") = type_selection;
  options.set("p1_type").doc() =
      "First parameter type. Options are: " + type_selection.candidates_str();

  options.set_parameter<CrossRef<Scalar>>("p2");
  options.set("p2").doc() = "Second parameter";

  options.set<EnumSelection>("p2_type") = type_selection;
  options.set("p2_type").doc() =
      "Second parameter type. Options are: " + type_selection.candidates_str();

  return options;
}

IsotropicElasticityTensor::IsotropicElasticityTensor(const OptionSet & options)
  : ElasticityTensor(options),
    _p1(declare_parameter<Scalar>("p1", "p1", /*allow nonlinear=*/true)),
    _p1_type(options.get<EnumSelection>("p1_type").as<ParamType>()),
    _p2(declare_parameter<Scalar>("p2", "p2", /*allow nonlinear=*/true)),
    _p2_type(options.get<EnumSelection>("p2_type").as<ParamType>())
{
}

void
IsotropicElasticityTensor::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "IsotropicElasticityTensor doesn't implement second derivatives.");

  const auto [lambda, dl_dp1, dl_dp2] = convert_to_lambda();
  const auto [mu, dm_dp1, dm_dp2] = convert_to_mu();

  auto Iv = SSR4::identity_vol(options());
  auto Is = SSR4::identity_sym(options());

  if (out)
    _p = 3.0 * lambda * Iv + 2.0 * mu * Is;

  if (dout_din)
  {
    if (const auto * const p1 = nl_param("p1"))
      _p.d(*p1) = 3.0 * dl_dp1 * Iv + 2.0 * dm_dp1 * Is;

    if (const auto * const p2 = nl_param("p2"))
      _p.d(*p2) = 3.0 * dl_dp2 * Iv + 2.0 * dm_dp2 * Is;
  }
}

std::tuple<Scalar, Scalar, Scalar>
IsotropicElasticityTensor::convert_to_lambda()
{
  if ((_p1_type == ParamType::YOUNGS) && (_p2_type == ParamType::POISSONS))
    return std::make_tuple(_p1 * _p2 / ((1 + _p2) * (1 - 2 * _p2)),
                           -_p2 / (2 * _p2 * _p2 + _p2 - 1),
                           (_p1 + 2 * _p1 * _p2 * _p2) /
                               ((2 * _p2 * _p2 + _p2 - 1) * (2 * _p2 * _p2 + _p2 - 1)));
  else if ((_p1_type == ParamType::POISSONS) && (_p2_type == ParamType::YOUNGS))
    return std::make_tuple(_p2 * _p1 / ((1 + _p1) * (1 - 2 * _p1)),
                           (_p2 + 2 * _p2 * _p1 * _p1) /
                               ((2 * _p1 * _p1 + _p1 - 1) * (2 * _p1 * _p1 + _p1 - 1)),
                           -_p1 / (2 * _p1 * _p1 + _p1 - 1));
  else
    throw NEMLException("Unsupported combination of input parameter types: " +
                        std::string(input_options().get<EnumSelection>("p1_type")) + " and " +
                        std::string(input_options().get<EnumSelection>("p2_type")));
}

std::tuple<Scalar, Scalar, Scalar>
IsotropicElasticityTensor::convert_to_mu()
{
  if ((_p1_type == ParamType::YOUNGS) && (_p2_type == ParamType::POISSONS))
    return std::make_tuple(
        _p1 / (2 * (1 + _p2)), 1.0 / (2.0 * 2 * _p2), -_p1 / (2 * (1 + _p2) * (1 + _p2)));
  else if ((_p1_type == ParamType::POISSONS) && (_p2_type == ParamType::YOUNGS))
    return std::make_tuple(
        _p2 / (2 * (1 + _p1)), -_p2 / (2 * (1 + _p1) * (1 + _p1)), 1 / (2 + 2 * _p1));
  else
    throw NEMLException("Unsupported combination of input parameter types: " +
                        std::string(input_options().get<EnumSelection>("p1_type")) + " and " +
                        std::string(input_options().get<EnumSelection>("p2_type")));
}

} // namespace neml2
