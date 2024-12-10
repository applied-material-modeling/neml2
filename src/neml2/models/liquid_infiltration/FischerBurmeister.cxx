#include "neml2/models/liquid_infiltration/FischerBurmeister.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(FischerBurmeister);
OptionSet
FischerBurmeister::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("Condition_A") = VariableName("state", "a");
  options.set<VariableName>("Condition_B") = VariableName("state", "b");
  options.set<VariableName>("Fischer_Burmeister") = VariableName("state", "fb");

  return options;
}

FischerBurmeister::FischerBurmeister(const OptionSet & options)
  : Model(options),
    _a(declare_input_variable<Scalar>("Condition_A")),
    _b(declare_input_variable<Scalar>("Condition_B")),
    _fb(declare_output_variable<Scalar>("Fischer_Burmeister"))
{
}

void
FischerBurmeister::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    // std::cout << "_a = " << _a.value().item<Real>() << std::endl;
    // std::cout << "_b = " << _b.value().item<Real>() << std::endl;
    _fb = _a + _b - math::sqrt(_a * _a + _b * _b);
    // std::cout << "_fb = " << _fb.value().item<Real>() << std::endl;
  }

  if (dout_din)
  {
    _fb.d(_a) = 1.0 - _a / math::sqrt(_a * _a + _b * _b + machine_precision());
    _fb.d(_b) = 1.0 - _b / math::sqrt(_a * _a + _b * _b + machine_precision());
  }
}
}