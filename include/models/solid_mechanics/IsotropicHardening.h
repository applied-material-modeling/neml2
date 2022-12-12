#pragma once

#include "models/Model.h"

namespace neml2
{
class IsotropicHardening : public Model
{
public:
  IsotropicHardening(const std::string & name);

  IsotropicHardening(InputParameters & params);
};
} // namespace neml2
