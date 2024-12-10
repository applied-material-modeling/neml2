#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class LiquidProductDiffusion1D : public Model
{
public:
  static OptionSet expected_options(); // shared by all

  LiquidProductDiffusion1D(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // Parameter
  const Scalar & _rho_rat;
  const Scalar & _phi0;
  const Scalar & _M;
  const Scalar & _D;
  const Scalar & _lc;

  // State Variables
  const Variable<Scalar> & _r1;
  const Variable<Scalar> & _sqrtd;

  // Output
  Variable<Scalar> & _sqrtd_growth;
};
}