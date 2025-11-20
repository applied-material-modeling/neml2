#include "neml2/models/Model.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
// @begin:signature
class ProjectileAcceleration : public Model
{
public:
  static OptionSet expected_options();
  ProjectileAcceleration(const OptionSet & options);

protected:
  // Model forward operator to be defined in the following tutorials
  void set_value(bool, bool, bool) override {}
};
// @end:signature

// @begin:registration
register_NEML2_object(ProjectileAcceleration);
// @end:registration

// @begin:expected_options
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
// @end:expected_options

// @begin:constructor
ProjectileAcceleration::ProjectileAcceleration(const OptionSet & options)
  : Model(options)
// Variable and parameter declarations are to be added in the following tutorials
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
  auto model = neml2::load_model("input.i", "accel");

  // Print out each option parsed from the input file
  const auto & options = model->input_options();

  std::cout << "                  velocity: " << options.get("velocity") << std::endl;
  std::cout << "              acceleration: " << options.get("acceleration") << std::endl;
  std::cout << "gravitational_acceleration: " << options.get("gravitational_acceleration")
            << std::endl;
  std::cout << "         dynamic_viscosity: " << options.get("dynamic_viscosity") << std::endl;
}
// @end:main
