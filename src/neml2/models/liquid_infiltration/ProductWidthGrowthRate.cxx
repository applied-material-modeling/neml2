#include "neml2/models/liquid_infiltration/ProductWidthGrowthRate.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ProductWidthGrowthRate);
OptionSet
ProductWidthGrowthRate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("Thickness_Rate") = VariableName("state", "ddot");
  options.set<VariableName>("Scale") = VariableName("state", "scale");
  options.set<VariableName>("Ideal_Thickness_Growth") = VariableName("state", "rate");
  options.set<VariableName>("Switch") = VariableName("state", "switch");
  options.set<VariableName>("Residual_Delta") = VariableName("residual", "rdelta");

  return options;
}

ProductWidthGrowthRate::ProductWidthGrowthRate(const OptionSet & options)
  : Model(options),
    _ddot(declare_input_variable<Scalar>("Thickness_Rate")),
    _scale(declare_input_variable<Scalar>("Scale")),
    _rate(declare_input_variable<Scalar>("Ideal_Thickness_Growth")),
    _smooth(declare_input_variable<Scalar>("Switch")),
    _rdelta(declare_output_variable<Scalar>("Residual_Delta"))
{
}

void
ProductWidthGrowthRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    // std::cout << "_ddot = " << _ddot.value().item<Real>() << std::endl;
    // std::cout << "_scale = " << _scale.value().item<Real>() << std::endl;
    // std::cout << "_rate = " << _rate.value().item<Real>() << std::endl;
    // std::cout << "_smooth = " << _smooth.value().item<Real>() << std::endl;

    _rdelta = _ddot - _scale * _rate * _smooth;
    // std::cout << "_rdelta = " << _rdelta.value().item<Real>() << std::endl;
  }

  if (dout_din)
  {
    _rdelta.d(_ddot) = neml2::Scalar::full(1.0);
    _rdelta.d(_scale) = -_rate * _smooth;
    _rdelta.d(_rate) = -_scale * _smooth;
    _rdelta.d(_smooth) = -_scale * _rate;
  }
}
}