#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class InletGap : public Model
{
public:
  static OptionSet expected_options(); // shared by all

  InletGap(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // Parameter
  const Scalar & _M;
  const Scalar & _phi0;
  // State Variables
  const Variable<Scalar> & _sqrtd;

  // Residual Variables
  // Variable<Vec> & _residual;
  Variable<Scalar> & _r1;
};
}