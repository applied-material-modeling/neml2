#include "ex1.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
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
  : Model(options)
// Variable and parameter declarations are to be added in the following tutorials
{
}
}

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
