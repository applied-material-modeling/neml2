#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class ChemMassBalance : public Model
{
public:
  static OptionSet expected_options(); // shared by all

  ChemMassBalance(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // State Variables
  const Variable<Scalar> & _in;
  const Variable<Scalar> & _switch;
  const Variable<Scalar> & _mreact;
  const Variable<Scalar> & _current;

  // Residual Variables
  Variable<Scalar> & _total;
};
}