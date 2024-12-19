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

#include "neml2/models/liquid_infiltration/ChemMassBalance.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ChemMassBalance);
OptionSet
ChemMassBalance::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Chemical mass balance of the infiltrated model: \\f$ total = current - "
                  "switch * in + minus_reaction \\f$";

  options.set_input("in") = VariableName("state", "in");
  options.set("in").doc() = "Mass rate in.";

  options.set_input("switch") = VariableName("state", "switch");
  options.set("switch").doc() = "Switching to take care of the mass rate out";

  options.set_input("minus_reaction") = VariableName("state", "mreact");
  options.set("minus_reaction").doc() = "negative of mass reaction rate.";

  options.set_input("current") = VariableName("state", "current");
  options.set("current").doc() = "The current mass rate in the system";

  options.set_output("total") = VariableName("residual", "total");
  options.set("total").doc() = "Chemical mass balance of the infiltrated model.";

  return options;
}

ChemMassBalance::ChemMassBalance(const OptionSet & options)
  : Model(options),
    _in(declare_input_variable<Scalar>("in")),
    _switch(declare_input_variable<Scalar>("switch")),
    _mreact(declare_input_variable<Scalar>("minus_reaction")),
    _current(declare_input_variable<Scalar>("current")),
    _total(declare_output_variable<Scalar>("total"))
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