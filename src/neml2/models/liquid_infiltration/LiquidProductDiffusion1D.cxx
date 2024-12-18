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

#include "neml2/models/liquid_infiltration/LiquidProductDiffusion1D.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(LiquidProductDiffusion1D);
OptionSet
LiquidProductDiffusion1D::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Product thickness growth based on the 1D Diffusion equation of liquid through "
                  "the product to react with the solid.";

  options.set_parameter<CrossRef<Scalar>>("liquid_product_density_ratio");
  options.set("liquid_product_density_ratio").doc() =
      "The ratio between the density of the liquid and the density of the product.";

  options.set_parameter<CrossRef<Scalar>>("initial_porosity");
  options.set("initial_porosity").doc() =
      "Initial porosity of the RVE in the absence of product and liquid.";

  options.set_parameter<CrossRef<Scalar>>("product_thickness_growth_ratio");
  options.set("product_thickness_growth_ratio").doc() =
      "Volume expansion ratio towards the liquid from the product - solid reactions."
      "Value should be between 0 and 1. For Si + C -> SiC, set this to 0.576.";

  options.set_parameter<CrossRef<Scalar>>("liquid_product_diffusion_coefficient");
  options.set("liquid_product_diffusion_coefficient").doc() =
      "The diffusion coefficient of the liquid through the product.";

  options.set_parameter<CrossRef<Scalar>>("representative_pores_size");
  options.set("representative_pores_size").doc() = "Representative pores size of the RVE.";

  options.set_input("inlet_gap") = VariableName("state", "r1");
  options.set("inlet_gap").doc() = "The width of the RVE's inlet.";

  options.set_input("product_thickness") = VariableName("state", "delta");
  options.set("product_thickness").doc() = "Thickness of the product in the RVE.";

  options.set_output("ideal_thickness_growth") = VariableName("state", "delta_growth");
  options.set("ideal_thickness_growth").doc() = "Ideal's product's thickness growth.";

  return options;
}

LiquidProductDiffusion1D::LiquidProductDiffusion1D(const OptionSet & options)
  : Model(options),
    _rho_rat(declare_parameter<Scalar>(
        "rho_rat", "liquid_product_density_ratio", /*allow_nonlinear=*/true)),
    _phi0(declare_parameter<Scalar>("phi0", "initial_porosity")),
    _M(declare_parameter<Scalar>("M", "product_thickness_growth_ratio")),
    _D(declare_parameter<Scalar>(
        "D", "liquid_product_diffusion_coefficient", /*allow_nonlinear=*/true)),
    _lc(declare_parameter<Scalar>("lc", "representative_pores_size")),
    _r1(declare_input_variable<Scalar>("inlet_gap")),
    _sqrtd(declare_input_variable<Scalar>("product_thickness")),
    _sqrtd_growth(declare_output_variable<Scalar>("ideal_thickness_growth"))
{
}

void
LiquidProductDiffusion1D::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto Dbar = _D / (_lc * _lc);
  auto denom =
      2.0 * _sqrtd * _sqrtd * _sqrtd * (math::sqrt(_phi0) + (1.0 - 2.0 * _M) * _sqrtd * _sqrtd);

  if (out)
  {
    _sqrtd_growth = Dbar * _rho_rat * _r1 / denom;
  }

  if (dout_din)
  {
    const auto * const rho_rat = nl_param("rho_rat");
    const auto * const D = nl_param("D");

    if (rho_rat)
      _sqrtd_growth.d(*rho_rat) = Dbar * _r1 / denom;

    if (D)
      _sqrtd_growth.d(*D) = _rho_rat * _r1 / denom / (_lc * _lc);

    _sqrtd_growth.d(_r1) = Dbar * _rho_rat / denom;
    _sqrtd_growth.d(_sqrtd) =
        -(_D * _r1 * _rho_rat *
          (5.0 * _sqrtd * _sqrtd - 10.0 * _M * _sqrtd * _sqrtd + 3.0 * math::sqrt(_phi0))) /
        (2.0 * _sqrtd * _sqrtd * _sqrtd * _sqrtd * _lc * _lc *
         (_sqrtd * _sqrtd - 2.0 * _M * _sqrtd * _sqrtd + math::sqrt(_phi0)) *
         (_sqrtd * _sqrtd - 2.0 * _M * _sqrtd * _sqrtd + math::sqrt(_phi0)));
  }
}
}