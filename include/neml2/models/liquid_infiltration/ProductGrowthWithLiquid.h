#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class ProductGrowthWithLiquid : public Model
{
public:
  static OptionSet expected_options(); // shared by all

  ProductGrowthWithLiquid(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // Parameter
  const Scalar & _oL;

  // State Variables
  const Variable<Scalar> & _r1;
  const Variable<Scalar> & _h;
  const Variable<Scalar> & _alpha;

  // Residual Variables
  // Variable<Vec> & _residual;
  Variable<Scalar> & _pcond;
};
}