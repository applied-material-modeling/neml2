#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class SwitchingFunction : public Model
{
public:
  static OptionSet expected_options(); // shared by all

  SwitchingFunction(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // Parameter
  const Scalar & _nn;

  // State Variables
  const Variable<Scalar> & _var;

  // Variable<Vec> & _residual;
  Variable<Scalar> & _smooth;
};
}