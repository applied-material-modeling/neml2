// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/user_tensors/UserTensorBase.h"

namespace neml2
{
template <class T>
OptionSet
UserTensorBase<T>::expected_options()
{
  OptionSet options = NEML2Object::expected_options();
  options.section() = "Tensors";

  MultiEnumSelection shape_manip_types({"dynamic_expand",
                                        "dynamic_reshape",
                                        "dynamic_squeeze",
                                        "dynamic_unsqueeze",
                                        "dynamic_transpose",
                                        "dynamic_movedim",
                                        "dynamic_flatten",
                                        "intmd_expand",
                                        "intmd_reshape",
                                        "intmd_squeeze",
                                        "intmd_unsqueeze",
                                        "intmd_transpose",
                                        "intmd_movedim",
                                        "intmd_flatten",
                                        "INVALID"},
                                       {"INVALID"});
  options.set<MultiEnumSelection>("shape_manipulations") = shape_manip_types;
  options.set("shape_manipulations").doc() = "A list of shape manipulation operations to apply to "
                                             "the created tensor. Supported operations are: " +
                                             shape_manip_types.join();

  options.set<std::vector<TensorShape>>("shape_manipulation_args") = {};
  options.set("shape_manipulation_args").doc() =
      "A list of arguments corresponding to each shape manipulation operation. The number of "
      "entries should match the number of operations in 'shape_manipulations'. Each entry is a "
      "tensor shape that encodes the arguments for the corresponding operation. For operations "
      "that do not require any argument, an empty shape, i.e. (), should be used.";

  return options;
}

template <class T>
UserTensorBase<T>::UserTensorBase(const OptionSet & options)
  : NEML2Object(options),
    T(),
    _manips(options.get<MultiEnumSelection>("shape_manipulations")),
    _manip_args(options.get<std::vector<TensorShape>>("shape_manipulation_args"))
{
  if (options.user_specified("shape_manipulations") !=
      options.user_specified("shape_manipulation_args"))
    throw FactoryException(
        "Options 'shape_manipulations' and 'shape_manipulation_args' must be specified together.");
  if (options.user_specified("shape_manipulations") &&
      options.user_specified("shape_manipulation_args"))
    if (_manips.size() != _manip_args.size())
      throw FactoryException("The number of shape manipulation operations (" +
                             std::to_string(_manips.size()) +
                             ") does not match the number of shape manipulation arguments (" +
                             std::to_string(_manip_args.size()) + ").");
}

template <class T>
std::string
UserTensorBase<T>::tensor_type()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 7 chars to remove 'neml2::'
  return utils::demangle(typeid(T).name()).substr(7);
}

template <std::size_t... Expected>
static void
assert_arg_dim(const std::string & op, const TensorShape & arg)
{
  std::string expected_str;
  std::size_t i = 0;
  ((expected_str += (i++ ? " or " : "") + std::to_string(Expected)), ...);
  neml_assert(((arg.size() == Expected) || ...),
              "Shape manipulation operation '",
              op,
              "' expects argument of dimension ",
              expected_str,
              ", but got argument ",
              arg);
}

static void
apply_shape_manipulation(Tensor & tensor, const std::string & op, const TensorShape & arg)
{
  if (op == "dynamic_expand")
    tensor = tensor.dynamic_expand(arg);
  else if (op == "dynamic_reshape")
    tensor = tensor.dynamic_reshape(arg);
  else if (op == "dynamic_squeeze")
  {
    assert_arg_dim<1>("dynamic_squeeze", arg);
    tensor = tensor.dynamic_squeeze(arg[0]);
  }
  else if (op == "dynamic_unsqueeze")
  {
    assert_arg_dim<1, 2>("dynamic_unsqueeze", arg);
    if (arg.size() == 1)
      tensor = tensor.dynamic_unsqueeze(arg[0]);
    else
      tensor = tensor.dynamic_unsqueeze(arg[0], arg[1]);
  }
  else if (op == "dynamic_transpose")
  {
    assert_arg_dim<2>("dynamic_transpose", arg);
    tensor = tensor.dynamic_transpose(arg[0], arg[1]);
  }
  else if (op == "dynamic_movedim")
  {
    assert_arg_dim<2>("dynamic_movedim", arg);
    tensor = tensor.dynamic_movedim(arg[0], arg[1]);
  }
  else if (op == "dynamic_flatten")
  {
    assert_arg_dim<0>("dynamic_flatten", arg);
    tensor = tensor.dynamic_flatten();
  }
  else if (op == "intmd_expand")
    tensor = tensor.intmd_expand(arg);
  else if (op == "intmd_reshape")
    tensor = tensor.intmd_reshape(arg);
  else if (op == "intmd_squeeze")
  {
    assert_arg_dim<1>("intmd_squeeze", arg);
    tensor = tensor.intmd_squeeze(arg[0]);
  }
  else if (op == "intmd_unsqueeze")
  {
    assert_arg_dim<1, 2>("intmd_unsqueeze", arg);
    if (arg.size() == 1)
      tensor = tensor.intmd_unsqueeze(arg[0]);
    else
      tensor = tensor.intmd_unsqueeze(arg[0], arg[1]);
  }
  else if (op == "intmd_transpose")
  {
    assert_arg_dim<2>("intmd_transpose", arg);
    tensor = tensor.intmd_transpose(arg[0], arg[1]);
  }
  else if (op == "intmd_movedim")
  {
    assert_arg_dim<2>("intmd_movedim", arg);
    tensor = tensor.intmd_movedim(arg[0], arg[1]);
  }
  else if (op == "intmd_flatten")
  {
    assert_arg_dim<0>("intmd_flatten", arg);
    tensor = tensor.intmd_flatten();
  }
  else
    throw NEMLException("Unknown shape manipulation operation: " + op);
}

template <class T>
void
UserTensorBase<T>::setup()
{
  NEML2Object::setup();

  auto tensor = neml2::Tensor(make());

  if (input_options().user_specified("shape_manipulations"))
    for (std::size_t i = 0; i < _manips.size(); i++)
      apply_shape_manipulation(tensor, _manips.selections()[i], _manip_args[i]);

  *static_cast<T *>(this) = T(tensor);
}

#define INSTANTIATE_USERTENSORBASE(T) template class UserTensorBase<T>
FOR_ALL_TENSORBASE(INSTANTIATE_USERTENSORBASE);

} // namespace neml2
