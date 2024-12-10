#include "neml2/models/liquid_infiltration/SwitchingFunction.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(SwitchingFunction);
OptionSet
SwitchingFunction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<CrossRef<Scalar>>("Smooth_Degree");
  options.set<VariableName>("variable") = VariableName("state", "var");
  options.set<VariableName>("switch_out") = VariableName("state", "out");

  return options;
}

SwitchingFunction::SwitchingFunction(const OptionSet & options)
  : Model(options),
    _nn(declare_parameter<Scalar>("nn", "Smooth_Degree")),
    _var(declare_input_variable<Scalar>("variable")),
    _smooth(declare_output_variable<Scalar>("switch_out"))
{
}

void
SwitchingFunction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _smooth = 1.0 - math::sigmoid(_var - 1.0, _nn);
  }

  if (dout_din)
  {
    _smooth.d(_var) =
        1.0 / 2.0 * (_nn * (math::tanh(_nn * (_var - 1.0)) * math::tanh(_nn * (_var - 1.0)) - 1.0));
  }
}
}