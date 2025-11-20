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
  void set_value(bool, bool, bool) override;

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

ProjectileAcceleration::ProjectileAcceleration(const OptionSet & options)
  : Model(options),
    _v(declare_input_variable<Vec>("velocity")),
    _a(declare_output_variable<Vec>("acceleration")),
    _g(declare_buffer<Vec>("g", "gravitational_acceleration")),
    _mu(declare_parameter<Scalar>("mu", "dynamic_viscosity"))
{
}

void
ProjectileAcceleration::set_value(bool out, bool dout, bool /*d2out*/)
{
  if (out)
    _a = _g - _mu * _v;

  if (dout)
    _a.d(_v) = -_mu * Vec::identity_map(_v.options());
}
}
