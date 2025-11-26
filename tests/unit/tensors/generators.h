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

#include <vector>

#include "neml2/misc/types.h"
#include "neml2/tensors/Tensor.h"

namespace test
{
struct GeneratedTensorConfig
{
  GeneratedTensorConfig(neml2::Dtype, neml2::Device);

  // Pretty string description of the tensor configuration
  std::string desc() const;

  // Independent tensor configurations
  neml2::Dtype dtype;
  neml2::Device device;

  // Inferred tensor configurations
  neml2::TensorOptions options;
};

struct GeneratedTensorShape
{
  GeneratedTensorShape(neml2::TensorShapeRef, neml2::TensorShapeRef, neml2::TensorShapeRef);

  // Pretty string description of the tensor configuration
  std::string desc() const;

  // Independent tensor configurations
  neml2::TensorShape dynamic_sizes;
  neml2::TensorShape intmd_sizes;
  neml2::TensorShape base_sizes;

  // Inferred tensor configurations
  neml2::TensorShape sizes;
  neml2::TensorShape batch_sizes;
  neml2::TensorShape static_sizes;
  neml2::Size dim;
  neml2::Size dynamic_dim;
  neml2::Size intmd_dim;
  neml2::Size base_dim;
  neml2::Size batch_dim;
  neml2::Size static_dim;
};

std::vector<neml2::Dtype> int_dtypes();
std::vector<neml2::Dtype> fp_dtypes();
std::vector<neml2::Dtype> dtypes();
std::vector<neml2::Device> devices();
std::vector<neml2::TensorShape> shapes();

GeneratedTensorConfig generate_tensor_config(
    const std::optional<std::vector<neml2::Dtype>> & gen_dtypes = std::nullopt,
    const std::optional<std::vector<neml2::Device>> & gen_devices = std::nullopt);

template <typename T = neml2::Tensor>
GeneratedTensorShape generate_tensor_shape(
    const std::optional<std::vector<neml2::TensorShape>> & dynamic_shapes = std::nullopt,
    const std::optional<std::vector<neml2::TensorShape>> & intmd_shapes = std::nullopt,
    const std::optional<std::vector<neml2::TensorShape>> & base_shapes = std::nullopt);

bool match_tensor_config(const neml2::Tensor & tensor, const GeneratedTensorConfig & config);
bool match_tensor_shape(const neml2::Tensor & tensor, const GeneratedTensorShape & shape);

template <class T = neml2::ATensor>
T generate_random_tensor(const GeneratedTensorConfig &, const GeneratedTensorShape &);
template <class T = neml2::ATensor>
T generate_full_tensor(const GeneratedTensorConfig &,
                       const GeneratedTensorShape &,
                       const neml2::CScalar &);
} // namespace test
