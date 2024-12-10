#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class FischerBurmeister : public Model
{
public:
  static OptionSet expected_options(); // shared by all

  FischerBurmeister(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // Parameter
  const Variable<Scalar> & _a;
  const Variable<Scalar> & _b;

  // Residual Variables
  // Variable<Vec> & _residual;
  Variable<Scalar> & _fb;
};
}