#include "neml2/neml2.h"
#include "neml2/tensors/SR2.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto factory = load_input("input.i");
  auto eq1 = factory->get_model("eq1");
  auto eq2 = factory->get_model("eq2");
  auto eq3 = factory->get_model("eq3");

  // Create the input variables
  auto a_name = VariableName("forces", "a");
  auto b_name = VariableName("state", "b");
  auto a = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03);
  auto b = SR2::fill(100, 20, 10, 5, -30, -20);

  // Evaluate the first model to get a_bar
  auto a_bar_name = VariableName("state", "a_bar");
  auto a_bar = eq1->value({{a_name, a}})[a_bar_name];

  // Evaluate the second model to get b_bar
  auto b_bar_name = VariableName("state", "b_bar");
  auto b_bar = eq2->value({{b_name, b}})[b_bar_name];

  // Evaluate the third model to get b_rate
  eq3->set_parameter("c_0", b_bar);
  eq3->set_parameter("c_1", a_bar);
  auto b_rate_name = VariableName("state", "b_rate");
  auto b_rate = eq3->value({{a_name, a}, {b_name, b}})[b_rate_name];

  std::cout << "b_rate: \n" << b_rate << std::endl;
}
