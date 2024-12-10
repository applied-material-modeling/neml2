#include "neml2/models/liquid_infiltration/ProductGrowthWithLiquid.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ProductGrowthWithLiquid);
OptionSet
ProductGrowthWithLiquid::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<CrossRef<Scalar>>("Liquid_Molar_Volume");
  options.set<VariableName>("Inlet_Gap") = VariableName("state", "r1");
  options.set<VariableName>("Product_Height") = VariableName("state", "h");
  options.set<VariableName>("Liquid_Saturation") = VariableName("state", "alpha");
  options.set<VariableName>("Phi_Condition") = VariableName("state", "pcond");

  return options;
}

ProductGrowthWithLiquid::ProductGrowthWithLiquid(const OptionSet & options)
  : Model(options),
    _oL(declare_parameter<Scalar>("oL", "Liquid_Molar_Volume", /*allow_nonlinear=*/true)),
    _r1(declare_input_variable<Scalar>("Inlet_Gap")),
    _h(declare_input_variable<Scalar>("Product_Height")),
    _alpha(declare_input_variable<Scalar>("Liquid_Saturation")),
    _pcond(declare_output_variable<Scalar>("Phi_Condition"))
{
}

void
ProductGrowthWithLiquid::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _pcond = _r1 * _r1 * _h - _alpha * _oL;
  }

  if (dout_din)
  {
    const auto * const oL = nl_param("oL");
    if (oL)
      _pcond.d(*oL) = -_alpha;

    _pcond.d(_r1) = 2.0 * _r1 * _h;
    _pcond.d(_h) = _r1 * _r1;
    _pcond.d(_alpha) = -_oL;
  }
}
}