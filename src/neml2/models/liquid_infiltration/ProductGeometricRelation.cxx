#include "neml2/models/liquid_infiltration/ProductGeometricRelation.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ProductGeometricRelation);
OptionSet
ProductGeometricRelation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<CrossRef<Scalar>>("Product_Molar_Volume");
  options.set<VariableName>("Inlet_Gap") = VariableName("state", "r1");
  options.set<VariableName>("Product_Height") = VariableName("state", "h");
  options.set<VariableName>("Product_Thickness") = VariableName("state", "delta");
  options.set<VariableName>("Product_Saturation") = VariableName("state", "alphaP");

  return options;
}

ProductGeometricRelation::ProductGeometricRelation(const OptionSet & options)
  : Model(options),
    _oP(declare_parameter<Scalar>("oP", "Product_Molar_Volume", /*allow_nonlinear=*/true)),
    _r1(declare_input_variable<Scalar>("Inlet_Gap")),
    _sqrtd(declare_input_variable<Scalar>("Product_Thickness")),
    _h(declare_input_variable<Scalar>("Product_Height")),
    _alphaP(declare_output_variable<Scalar>("Product_Saturation"))
{
}

void
ProductGeometricRelation::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _alphaP =
        _h / _oP *
        (2.0 * _r1 * _sqrtd * _sqrtd +
         _sqrtd * _sqrtd * _sqrtd * _sqrtd); // can not get power to work, math::pow(_sqrtd, 4.0));
  }

  if (dout_din)
  {
    const auto * const oP = nl_param("oP");
    if (oP)
      _alphaP.d(*oP) =
          -_h / (_oP * _oP) * (2.0 * _r1 * _sqrtd * _sqrtd + _sqrtd * _sqrtd * _sqrtd * _sqrtd);

    _alphaP.d(_r1) = 2.0 * _h * _sqrtd * _sqrtd / _oP;
    _alphaP.d(_h) = (2.0 * _r1 * _sqrtd * _sqrtd + _sqrtd * _sqrtd * _sqrtd * _sqrtd) / _oP;
    _alphaP.d(_sqrtd) = (4 * _sqrtd * _h * (_sqrtd * _sqrtd + _r1)) / _oP;
  }
}
}