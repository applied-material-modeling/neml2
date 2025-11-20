#include <torch/cuda.h>
#include "neml2/models/Model.h"
#include "neml2/tensors/SR2.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto model = load_model("input.i", "my_model");

  // Pick the device
  auto device = torch::cuda::is_available() ? kCUDA : kCPU;

  // Send the model parameters to the device
  model->to(device);

  // Create the strain on the device
  auto strain_name = VariableName("forces", "E");
  auto strain = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device);

  // Evaluate the model
  auto output = model->value({{strain_name, strain}});

  // Get the stress back to CPU
  auto stress_name = VariableName("state", "S");
  auto stress = output[stress_name].to(kCPU);
}
