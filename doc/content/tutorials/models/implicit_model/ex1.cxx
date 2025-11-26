#include "neml2/models/Model.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto system = load_model("input1.i", "system");
  std::cout << *system << std::endl;
}
