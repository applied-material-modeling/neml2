#pragma once

#include "models/solid_mechanics/YieldFunction.h"

namespace neml2
{

/// Yield function with no hardening
class KinematicHardeningYieldFunction : public YieldFunction
{
public:
  KinematicHardeningYieldFunction(const std::string & name,
                                  const std::shared_ptr<StressMeasure> & sm,
                                  Scalar s0);
};

} // namespace neml2
