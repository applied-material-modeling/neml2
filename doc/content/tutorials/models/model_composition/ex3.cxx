#include "neml2/neml2.h"

int
main()
{
  using namespace neml2;

  auto factory = load_input("input_composed.i");
  auto eq1 = factory->get_model("eq1");
  auto eq2 = factory->get_model("eq2");
  auto eq3 = factory->get_model("eq3");
  auto eq = factory->get_model("eq");

  std::cout << "eq1:\n" << *eq1 << std::endl << std::endl;
  std::cout << "eq2:\n" << *eq2 << std::endl << std::endl;
  std::cout << "eq3:\n" << *eq3 << std::endl << std::endl;
  std::cout << "eq:\n" << *eq << std::endl << std::endl;
}
