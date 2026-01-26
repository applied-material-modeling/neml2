#include "neml2/neml2.h"
#include "neml2/tensors/Tensor.h"

int
main()
{
  using namespace neml2;

  auto model = load_model("input.i", "my_model");

  for (auto && [pname, pval] : model->named_parameters())
    std::cout << pname << ":\n" << Tensor(*pval) << std::endl;
}
