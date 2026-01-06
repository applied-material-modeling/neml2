#include "neml2/tensors/Tensor.h"

using namespace neml2;
using namespace indexing;

int
main()
{
  // Create a tensor with shape (; 4, 3) filled with zeros
  auto a = Tensor::zeros({4, 3});
  std::cout << "a =\n" << a << std::endl;

  // b is a view into the first row and the third row of a
  auto b = a.base_index({Slice(None, None, 2)});
  std::cout << "b =\n" << b << std::endl;

  // Modification in a is reflected in b
  a += 1.0;
  std::cout << "\nAfter first modification" << std::endl;
  std::cout << "a =\n" << a << std::endl;
  std::cout << "b =\n" << b << std::endl;

  // Modification in data viewed by b is reflected in a
  b += 1.0;
  std::cout << "\nAfter second modification" << std::endl;
  std::cout << "a =\n" << a << std::endl;
  std::cout << "b =\n" << b << std::endl;
}
