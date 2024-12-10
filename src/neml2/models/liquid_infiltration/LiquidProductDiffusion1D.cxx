#include "neml2/models/liquid_infiltration/LiquidProductDiffusion1D.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(LiquidProductDiffusion1D);
OptionSet
LiquidProductDiffusion1D::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<CrossRef<Scalar>>("Liquid_Product_Density_Ratio");
  options.set<CrossRef<Scalar>>("Initial_Porosity");
  options.set<CrossRef<Scalar>>("Product_Thickness_Growth_Ratio");
  options.set<CrossRef<Scalar>>("Liquid_Product_Diffusion_Coefficient");
  options.set<CrossRef<Scalar>>("Representative_Pores_Size");

  options.set<VariableName>("Inlet_Gap") = VariableName("state", "r1");
  options.set<VariableName>("Product_Thickness") = VariableName("state", "delta");

  options.set<VariableName>("Ideal_Thickness_Growth") = VariableName("state", "delta_growth");

  return options;
}

LiquidProductDiffusion1D::LiquidProductDiffusion1D(const OptionSet & options)
  : Model(options),
    _rho_rat(declare_parameter<Scalar>(
        "rho_rat", "Liquid_Product_Density_Ratio", /*allow_nonlinear=*/true)),
    _phi0(declare_parameter<Scalar>("phi0", "Initial_Porosity")),
    _M(declare_parameter<Scalar>("M", "Product_Thickness_Growth_Ratio")),
    _D(declare_parameter<Scalar>(
        "D", "Liquid_Product_Diffusion_Coefficient", /*allow_nonlinear=*/true)),
    _lc(declare_parameter<Scalar>("lc", "Representative_Pores_Size")),
    _r1(declare_input_variable<Scalar>("Inlet_Gap")),
    _sqrtd(declare_input_variable<Scalar>("Product_Thickness")),
    _sqrtd_growth(declare_output_variable<Scalar>("Ideal_Thickness_Growth"))
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