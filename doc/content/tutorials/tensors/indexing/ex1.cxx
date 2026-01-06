#include "neml2/tensors/Tensor.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);

  // Create a tensor with shape (; 5)
  auto a = Tensor::create({3, 4, 5, 6, 7}, 0);
  std::cout << "a.base[2] = " << a.base_index({2}).item<double>() << std::endl;
  std::cout << "a.base[-1] = " << a.base_index({-1}).item<double>() << std::endl;
  std::cout << "a.base[-2] = " << a.base_index({-2}).item<double>() << '\n' << std::endl;

  // Create a tensor with shape (4;)
  auto b = Tensor::create({7, 6, 5, 4}, 1);
  std::cout << "b.batch[2] = " << b.batch_index({2}).item<double>() << std::endl;
  std::cout << "b.batch[-1] = " << b.batch_index({-1}).item<double>() << std::endl;
  std::cout << "b.batch[-2] = " << b.batch_index({-2}).item<double>() << std::endl;
}
