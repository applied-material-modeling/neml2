#include "neml2/models/Model.h"

namespace neml2
{
class ProjectileAcceleration : public Model
{
public:
  static OptionSet expected_options();
  ProjectileAcceleration(const OptionSet & options);

protected:
  // Model forward operator to be defined in the following tutorials
  void set_value(bool, bool, bool) override {}
};
}
