#include "neml2/models/liquid_infiltration/ChemMassBalance.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ChemMassBalance);
OptionSet
ChemMassBalance::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("In") = VariableName("state", "in");
  options.set<VariableName>("Switch") = VariableName("state", "switch");
  options.set<VariableName>("Minus_Reaction") = VariableName("state", "mreact");
  options.set<VariableName>("Current") = VariableName("state", "current");
  options.set<VariableName>("Total") = VariableName("residual", "total");

  return options;
}

ChemMassBalance::ChemMassBalance(const OptionSet & options)
  : Model(options),
    _in(declare_input_variable<Scalar>("In")),
    _switch(declare_input_variable<Scalar>("Switch")),
    _mreact(declare_input_variable<Scalar>("Minus_Reaction")),
    _current(declare_input_variable<Scalar>("Current")),
    _total(declare_output_variable<Scalar>("Total"))
{
}

void
ChemMassBalance::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {

    // std::cout << "_current = " << _current.value().item<Real>() << std::endl;
    // std::cout << "_switch = " << _switch.value().item<Real>() << std::endl;
    // std::cout << "_in = " << _in.value().item<Real>() << std::endl;
    // std::cout << "_mreact = " << _mreact.value().item<Real>() << std::endl;

    _total = _current - _switch * _in + _mreact;
    // std::cout << "_total = " << _total.value().item<Real>() << std::endl;
  }

  if (dout_din)
  {
    _total.d(_switch) = -_in;
    _total.d(_mreact) = neml2::Scalar::full(1.0);
    _total.d(_current) = neml2::Scalar::full(1.0);

    if (_in.is_dependent())
      _total.d(_in) = -_switch;

    if (currently_solving_nonlinear_system())
      return;
  }
}
}