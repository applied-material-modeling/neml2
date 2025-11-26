#include "neml2/models/Model.h"
#include "neml2/tensors/Tensor.h"

int
main()
{
  using namespace neml2;

  auto model = load_model("input2.i", "eq");
  std::cout << *model << std::endl << std::endl;
  std::cout << "eq1_alpha:\n" << Tensor(model->get_parameter("eq1_alpha")) << std::endl;
}
