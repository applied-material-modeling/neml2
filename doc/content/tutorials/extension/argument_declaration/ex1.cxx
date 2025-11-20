#include "neml2/models/Model.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"

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

  // @begin:variables
  // Velocity of the projectile (input variable)
  const Variable<Vec> & _v;

  // Acceleration of the projectile (output variable)
  Variable<Vec> & _a;
  // @end:variables

  // @begin:parameters
  // Gravitational acceleration (parameter)
  const Vec & _g;

  // Dynamic viscosity of the medium (parameter)
  const Scalar & _mu;
  // @end:parameters
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

// @begin:constructor
ProjectileAcceleration::ProjectileAcceleration(const OptionSet & options)
  : Model(options),
    _v(declare_input_variable<Vec>("velocity")),
    _a(declare_output_variable<Vec>("acceleration")),
    _g(declare_buffer<Vec>("g", "gravitational_acceleration")),
    _mu(declare_parameter<Scalar>("mu", "dynamic_viscosity"))
{
}
// @end:constructor
}

// @begin:main
int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto model = load_model("input.i", "accel");
  std::cout << *model << std::endl;
}
// @end:main
