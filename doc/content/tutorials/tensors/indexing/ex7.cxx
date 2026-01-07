#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/functions/linspace.h"

int
main()
{
  using namespace neml2;
  using namespace indexing;
  set_default_dtype(kFloat64);

  // Create a tensor with shape (3, 4; 2)
  auto a0 = Tensor::create({{0, 1}, {2, 3}, {4, 5}}, 1);
  auto a1 = Tensor::create({{3, 4}, {5, 6}, {7, 8}}, 1);
  auto a = dynamic_linspace(a0, a1, 4, 1);

  std::cout << "a.batch[:2, -3:4] =\n"
            << a.batch_index({Slice(None, 2), Slice(-3, 4)}) << std::endl;
}
