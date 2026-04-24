#include "neml2/neml2.h"
#include "neml2/tensors/SR2.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto eq = load_model("input_composed.i", "eq");

  // Create the input variables
  auto a = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03);
  auto b = SR2::fill(100, 20, 10, 5, -30, -20);

  // Evaluate the composed model
  auto b_rate = eq->value({{"a", a}, {"b", b}})["b_rate"];

  std::cout << "b_rate: \n" << b_rate << std::endl;
}
