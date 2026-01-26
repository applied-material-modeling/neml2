#include "neml2/neml2.h"
#include "neml2/tensors/SR2.h"

int
main()
{
  using namespace neml2;

  set_default_dtype(kFloat64);
  auto model = load_model("input.i", "my_model");

  // Create the strain
  auto strain_name = VariableName("forces", "E");
  auto strain = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03);

  // Evaluate the model
  auto output = model->value({{strain_name, strain}});

  // Get the stress
  auto stress_name = VariableName("state", "S");
  auto & stress = output[stress_name];

  std::cout << "strain: \n" << strain << std::endl;
  std::cout << "stress: \n" << stress << std::endl;
}
