#include "neml2/neml2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/jit.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto model = load_model("input.i", "eq");

  // Create example input variables for tracing
  auto a = Scalar::full(1.0);
  auto b = Scalar::full(2.0);
  auto c = Scalar::full(3.0);
  auto t = Scalar::full(0.1);
  auto a_n = Scalar::full(0.0);
  auto b_n = Scalar::full(1.0);
  auto c_n = Scalar::full(2.0);
  auto t_n = Scalar::full(0.0);

  // Evaluate the model for the first time
  // This is when tracing takes place
  model->value({{"a", a},
                {"b", b},
                {"c", c},
                {"t", t},
                {"a~1", a_n},
                {"b~1", b_n},
                {"c~1", c_n},
                {"t~1", t_n}});

  utils::last_executed_optimized_graph()->dump();
}
