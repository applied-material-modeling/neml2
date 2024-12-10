#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class ProductThicknessLimit : public Model
{
public:
  static OptionSet expected_options(); // shared by all

  ProductThicknessLimit(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // Parameter
  const Scalar & _phi0;
  const Scalar & _M;

  // State Variables
  const Variable<Scalar> & _sqrtd;

  // Output
  Variable<Scalar> & _dra;
};
}