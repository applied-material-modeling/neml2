#pragma once

#include "neml2/models/solid_mechanics/StressMeasure.h"

namespace neml2
{
class J2StressMeasure : public StressMeasure
{
public:
  J2StressMeasure(const std::string & name);

protected:
  /// The value of the stress measure
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  /// The derivative of the stress measure wrt mandel stress
  virtual void set_dvalue(LabeledVector in,
                          LabeledMatrix dout_din,
                          LabeledTensor<1, 3> * d2out_din2 = nullptr) const;
};

} // namespace neml2
