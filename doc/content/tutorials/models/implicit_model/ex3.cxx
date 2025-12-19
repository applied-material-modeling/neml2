#include "neml2/neml2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto model = load_model("input2.i", "model");

  // Create input variables
  // Unspecified variables are assumed to be zero
  auto E = SR2::fill(0.01, 0.005, -0.001);
  auto t = Scalar::full(1);

  // Solve the implicit model
  auto outputs = model->value({{VariableName("forces", "E"), E}, {VariableName("forces", "t"), t}});

  // Get the solution
  std::cout << "\nPlastic strain:\n" << outputs[VariableName("state", "Ep")] << std::endl;
}
