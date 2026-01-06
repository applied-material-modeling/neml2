#include <torch/torch.h>
#include "neml2/tensors/Tensor.h"

int
main()
{
  // A raw torch tensor
  auto A = torch::rand({5, 3, 2, 8});
  std::cout << "Shape of A: " << A.sizes() << '\n' << std::endl;

  // Mark it with batch dimension of 2
  auto B = neml2::Tensor(A, 2);
  std::cout << "      Shape of B: " << B.sizes() << std::endl;
  std::cout << "Batch shape of B: " << B.batch_sizes() << std::endl;
  std::cout << " Base shape of B: " << B.base_sizes() << std::endl;
}
