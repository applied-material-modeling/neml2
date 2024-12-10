#include "neml2/models/liquid_infiltration/LiquidDeficientPowerScale.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(LiquidDeficientPowerScale);
OptionSet
LiquidDeficientPowerScale::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<CrossRef<Scalar>>("Liquid_Molar_Volume");
  options.set<CrossRef<Scalar>>("Power");
  options.set<VariableName>("Inlet_Gap") = VariableName("state", "r1");
  options.set<VariableName>("Product_Height") = VariableName("state", "h");
  options.set<VariableName>("Liquid_Saturation") = VariableName("state", "alpha");
  options.set<VariableName>("Scale") = VariableName("state", "scale");

  return options;
}

LiquidDeficientPowerScale::LiquidDeficientPowerScale(const OptionSet & options)
  : Model(options),
    _oL(declare_parameter<Scalar>("oL", "Liquid_Molar_Volume", /*allow_nonlinear=*/true)),
    _p(declare_parameter<Scalar>("p", "Power")),
    _r1(declare_input_variable<Scalar>("Inlet_Gap")),
    _h(declare_input_variable<Scalar>("Product_Height")),
    _alpha(declare_input_variable<Scalar>("Liquid_Saturation")),
    _scale(declare_output_variable<Scalar>("Scale"))
{
}

void
LiquidDeficientPowerScale::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto hL = _alpha * _oL / (_r1 * _r1 + machine_precision());
  if (out)
  {
    _scale = math::pow(math::macaulay(hL) / _h, _p);
  }

  if (dout_din)
  {
    const auto * const oL = nl_param("oL");
    if (oL)
      _scale.d(*oL) = (_p * math::pow(math::macaulay(hL) / _h, _p)) / _oL;

    _scale.d(_r1) = -(2.0 * _p * math::pow(math::macaulay(hL) / _h, _p)) / _r1;
    _scale.d(_h) = -(_p * math::pow(math::macaulay(hL) / _h, _p)) / _h;
    _scale.d(_alpha) = (_p * math::pow(math::macaulay(hL) / _h, _p)) / _alpha;
  }
}
}