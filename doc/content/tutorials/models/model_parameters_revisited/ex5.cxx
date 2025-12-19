#include "neml2/neml2.h"

int
main()
{
  using namespace neml2;

  auto model = load_model("input3.i", "eq");
  std::cout << *model << std::endl;
}
