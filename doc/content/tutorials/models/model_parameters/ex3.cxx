#include "neml2/models/Model.h"
#include "neml2/tensors/Tensor.h"

int
main()
{
  using namespace neml2;

  auto model = load_model("input.i", "my_model");

  auto & G = model->get_parameter("G");
  auto & K = model->get_parameter("K");

  std::cout << "G:\n" << Tensor(G) << std::endl;
  std::cout << "K:\n" << Tensor(K) << std::endl;
}
