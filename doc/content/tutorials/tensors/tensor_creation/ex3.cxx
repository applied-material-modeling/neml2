#include "neml2/tensors/SR2.h"

using namespace neml2;

int
main()
{
  auto A = SR2::zeros({5, 3}, {}, TensorOptions().dtype(kFloat32).device(kCPU));
}
