@insert-title:tutorials-tensors

NEML2 is equipped with its own tensor library. Currently, PyTorch (more specifically, ATen) is the only supported tensor backend in NEML2. Therefore, all tensor types in NEML2 directly inherit from `torch::Tensor`. In the future, support for other tensor backends may be added, but the public-facing interfaces will remain largely the same.

This set of tutorials introduce the basic API of NEML2's tensor library:

@insert-subsection-list
