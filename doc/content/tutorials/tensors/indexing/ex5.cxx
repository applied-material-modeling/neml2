#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/linspace.h"

int
main()
{
  using namespace neml2;
  using namespace indexing;
  set_default_dtype(kFloat64);

  // Create a tensor with shape (20;)
  auto a0 = Scalar::full(0.0);
  auto a1 = Scalar::full(19.0);
  auto a = dynamic_linspace(a0, a1, 20);

  std::cout << "Basic syntax: start:stop:step" << std::endl;
  std::cout << "a.batch[5:17:2] =\n" << a.batch_index({Slice(5, 17, 2)}) << '\n' << std::endl;

  std::cout << "Negative start and stop are counted backward" << std::endl;
  std::cout << "a.batch[-15:-3:2] =\n" << a.batch_index({Slice(-15, -3, 2)}) << '\n' << std::endl;

  std::cout << "start default to 0" << std::endl;
  std::cout << "a.batch[:17:3] =\n" << a.batch_index({Slice(None, 17, 3)}) << '\n' << std::endl;

  std::cout << "stop default to 'consuming all remaining elements'" << std::endl;
  std::cout << "a.batch[12::2] =\n" << a.batch_index({Slice(12, None, 2)}) << '\n' << std::endl;

  std::cout << "step default to 1" << std::endl;
  std::cout << "a.batch[3:6:] =\n" << a.batch_index({Slice(3, 6, None)}) << '\n' << std::endl;

  std::cout << "Trailing colon(s) can be omitted" << std::endl;
  std::cout << "a.batch[3:6] =\n" << a.batch_index({Slice(3, 6)}) << '\n' << std::endl;
  std::cout << "a.batch[17:] =\n" << a.batch_index({Slice(17)}) << '\n' << std::endl;

  std::cout << "The default is therefore equivalent to slicing the entire dimension" << std::endl;
  std::cout << "a.batch[:] =\n" << a.batch_index({Slice()}) << std::endl;
}
