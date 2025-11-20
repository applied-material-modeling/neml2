#include "neml2/models/Model.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Scalar.h"

int
main()
{
  using namespace neml2;

  auto model = load_model("input.i", "my_model");

  // Before modification
  auto & G = model->get_parameter("G");
  std::cout << "G (before modification):\n" << Tensor(G) << std::endl;

  // After modification
  G = Scalar::full(59000);
  std::cout << "G (after modification):\n" << Tensor(G) << std::endl;
}
