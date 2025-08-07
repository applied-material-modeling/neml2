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

#pragma once

#include "neml2/user_tensors/UserTensorBase.h"
#include <filesystem>

namespace neml2
{
/**
 * @brief Read the tensors provided by a torch script file
 */
template <class T>
class FromTorchScript : public UserTensorBase<T>
{
public:
  static OptionSet expected_options();

  FromTorchScript(const OptionSet & options);

protected:
  /// Helper to make the tensor given user input options
  T make() const override;

private:
  /// Path to the torch script file
  const std::filesystem::path _torch_script;

  /// Name of the tensor to extract
  const std::string _tensor_name;

  /// Batch shape
  const TensorShape _batch_sizes;

  /// Base shape
  const TensorShape _base_sizes;

  /// Number of intermediate dimensions
  const unsigned int _intmd_dim;
};
} // namespace neml2
