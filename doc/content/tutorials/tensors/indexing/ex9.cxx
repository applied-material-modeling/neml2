#include <torch/torch.h>
#include "neml2/tensors/Tensor.h"

int
main()
{
  using namespace neml2;
  using namespace indexing;
  set_default_dtype(kFloat64);

  // Create a tensor with shape (5, 3, 1, 3; 1, 7, 8)
  auto a = Tensor(torch::rand({5, 3, 1, 3, 1, 7, 8}), 4);

  // Batch indexing with ellipsis
  std::cout << "a.batch[2:, ..., :].shape = "
            << a.batch_index({Slice(2), Ellipsis, Slice()}).sizes() << std::endl;

  // Batch indexing with none
  std::cout << "a.batch[2:, :, :, None, :].shape = "
            << a.batch_index({Slice(2), Slice(), Slice(), None, Slice()}).sizes() << std::endl;

  // Batch indexing with both ellipsis and none
  std::cout << "a.batch[..., None, :].shape = " << a.batch_index({Ellipsis, None, Slice()}).sizes()
            << std::endl;

  // Same rules apply to base indexing
  std::cout << "a.base[..., None, :].shape = " << a.base_index({Ellipsis, None, Slice()}).sizes()
            << std::endl;
}
