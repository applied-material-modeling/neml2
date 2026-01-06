#include "neml2/tensors/Tensor.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);

  // Create a tensor with shape (2, 2 ; 3, 1)
  auto a = Tensor::create(
      {{{{1}, {2}, {3}}, {{4}, {5}, {6}}}, {{{-1}, {-2}, {-3}}, {{-4}, {-5}, {-6}}}}, 2);

  // Single element indexing along batch dimensions
  std::cout << "a.batch[1, 0] = \n" << a.batch_index({1, 0}) << '\n' << std::endl;
  std::cout << "a.batch[0] = \n" << a.batch_index({0}) << '\n' << std::endl;

  // Single element indexing along base dimensions
  std::cout << "a.base[2, 0] = \n" << a.base_index({2, 0}) << '\n' << std::endl;
  std::cout << "a.base[1] = \n" << a.base_index({1}) << std::endl;
}
