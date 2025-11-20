#include "neml2/models/Model.h"

int
main()
{
  using namespace neml2;

  auto model = load_model("input.i", "my_model");
  std::cout << *model << std::endl;
}
