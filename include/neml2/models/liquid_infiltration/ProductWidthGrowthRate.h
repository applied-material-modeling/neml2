#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class ProductWidthGrowthRate : public Model
{
public:
  static OptionSet expected_options(); // shared by all

  ProductWidthGrowthRate(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // State Variables
  const Variable<Scalar> & _ddot;
  const Variable<Scalar> & _scale;
  const Variable<Scalar> & _rate;
  const Variable<Scalar> & _smooth;

  // Residual Variables
  Variable<Scalar> & _rdelta;
};
}