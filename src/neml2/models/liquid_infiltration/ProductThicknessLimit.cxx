#include "neml2/models/liquid_infiltration/ProductThicknessLimit.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ProductThicknessLimit);
OptionSet
ProductThicknessLimit::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<CrossRef<Scalar>>("Initial_Porosity");
  options.set<CrossRef<Scalar>>("Product_Thickness_Growth_Ratio");

  options.set<VariableName>("Product_Thickness") = VariableName("state", "delta");
  options.set<VariableName>("Limit_Ratio") = VariableName("state", "dratio");

  return options;
}

ProductThicknessLimit::ProductThicknessLimit(const OptionSet & options)
  : Model(options),
    _phi0(declare_parameter<Scalar>("phi0", "Initial_Porosity")),
    _M(declare_parameter<Scalar>("M", "Product_Thickness_Growth_Ratio")),
    _sqrtd(declare_input_variable<Scalar>("Product_Thickness")),
    _dra(declare_output_variable<Scalar>("Limit_Ratio"))
{
}

void
ProductThicknessLimit::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto limit1 = (1.0 - math::sqrt(_phi0)) / (1.0 - _M + machine_precision());
  auto limit2 = math::sqrt(_phi0) / (_M + machine_precision());
  auto dlimit = math::where(limit1 < limit2, limit1, limit2);

  if (out)
  {
    _dra = _sqrtd * _sqrtd / (dlimit + machine_precision());
  }

  if (dout_din)
  {
    _dra.d(_sqrtd) = 2.0 * _sqrtd / (dlimit + machine_precision());
  }
}
}