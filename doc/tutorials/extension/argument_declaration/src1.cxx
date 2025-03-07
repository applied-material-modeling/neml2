#include "neml2/models/Model.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
class ProjectileAcceleration : public Model
{
public:
  static OptionSet expected_options();
  ProjectileAcceleration(const OptionSet & options);

protected:
  void set_value(bool, bool, bool) override {}

  /// Velocity of the projectile (input variable)
  const Variable<Vec> & _v;

  /// Acceleration of the projectile (output variable)
  Variable<Vec> & _a;

  /// Gravitational acceleration (parameter)
  const Vec & _g;

  /// Dynamic viscosity of the medium (parameter)
  const Scalar & _mu;
};

register_NEML2_object(ProjectileAcceleration);
}

namespace neml2
{
OptionSet
ProjectileAcceleration::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("velocity");
  options.set<VariableName>("acceleration");
  options.set<TensorName<Vec>>("gravitational_acceleration");
  options.set<TensorName<Scalar>>("dynamic_viscosity");
  return options;
}
}
