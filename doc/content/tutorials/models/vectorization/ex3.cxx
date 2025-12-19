#include <torch/cuda.h>
#include "neml2/neml2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/functions/linspace.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);

  // Preparation
  Size N = 10;
  auto device = kCPU;
  auto model = load_model("input.i", "my_model");
  model->to(device);

  // Create the strain on the device
  auto strain_name = VariableName("forces", "E");
  auto strain_min = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device);
  auto strain_max = SR2::fill(0.5, 0.4, -0.2, 0.2, 0.3, 0.1, device);
  auto strain = dynamic_linspace(strain_min, strain_max, N);

  // Evaluate the model N times
  auto output = model->value({{strain_name, strain}});
}
