#include "neml2/models/Model.h"
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
  model->value({{VariableName("state", "a"), a},
                {VariableName("state", "b"), b},
                {VariableName("state", "c"), c},
                {VariableName("forces", "t"), t},
                {VariableName("old_state", "a"), a_n},
                {VariableName("old_state", "b"), b_n},
                {VariableName("old_state", "c"), c_n},
                {VariableName("old_forces", "t"), t_n}});

  utils::last_executed_optimized_graph()->dump();
}
