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

#include "unit/tensors/generators.h"

#include <c10/core/ScalarType.h>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "neml2/misc/errors.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/macros.h"
#include "neml2/tensors/tensors.h"

#include "utils.h"

namespace test
{
GeneratedTensorConfig::GeneratedTensorConfig(neml2::Dtype dtype, neml2::Device device)
  : dtype(dtype),
    device(device),
    options(neml2::TensorOptions().dtype(dtype).device(device))
{
}

std::string
GeneratedTensorConfig::desc() const
{
  std::stringstream ss;
  ss << "dtype=" << dtype << " device=" << device;
  return ss.str();
}

GeneratedTensorShape::GeneratedTensorShape(neml2::TensorShapeRef dynamic_sizes,
                                           neml2::TensorShapeRef intmd_sizes,
                                           neml2::TensorShapeRef base_sizes)
  : dynamic_sizes(dynamic_sizes.begin(), dynamic_sizes.end()),
    intmd_sizes(intmd_sizes.begin(), intmd_sizes.end()),
    base_sizes(base_sizes.begin(), base_sizes.end()),
    sizes(neml2::utils::add_shapes(dynamic_sizes, intmd_sizes, base_sizes)),
    batch_sizes(neml2::utils::add_shapes(dynamic_sizes, intmd_sizes)),
    static_sizes(neml2::utils::add_shapes(intmd_sizes, base_sizes)),
    dim(neml2::Size(sizes.size())),
    dynamic_dim(neml2::Size(dynamic_sizes.size())),
    intmd_dim(neml2::Size(intmd_sizes.size())),
    base_dim(neml2::Size(base_sizes.size())),
    batch_dim(dynamic_dim + intmd_dim),
    static_dim(intmd_dim + base_dim)
{
}

std::string
GeneratedTensorShape::desc() const
{
  std::stringstream ss;
  ss << "shape=(";

  for (std::size_t i = 0; i < dynamic_sizes.size(); ++i)
    ss << (i == 0 ? "" : ",") << dynamic_sizes[i];
  ss << ";";

  for (std::size_t i = 0; i < intmd_sizes.size(); ++i)
    ss << (i == 0 ? "" : ",") << intmd_sizes[i];
  ss << ";";

  for (std::size_t i = 0; i < base_sizes.size(); ++i)
    ss << (i == 0 ? "" : ",") << base_sizes[i];

  ss << ")";
  return ss.str();
}

std::vector<neml2::Dtype>
int_dtypes()
{
  return {neml2::kInt8, neml2::kInt16, neml2::kInt32, neml2::kInt64};
}

std::vector<neml2::Dtype>
fp_dtypes()
{
  return {neml2::kFloat16, neml2::kFloat32, neml2::kFloat64};
}

std::vector<neml2::Dtype>
dtypes()
{
  auto types = int_dtypes();
  auto fp_types = fp_dtypes();
  types.insert(types.end(), fp_types.begin(), fp_types.end());
  return types;
}

std::vector<neml2::Device>
devices()
{
  std::vector<neml2::Device> devs{neml2::kCPU};
  devs.insert(devs.end(),
              get_test_suite_additional_devices().begin(),
              get_test_suite_additional_devices().end());
  return devs;
}

std::vector<neml2::TensorShape>
shapes()
{
  return {neml2::TensorShape{}, neml2::TensorShape{3, 1}};
}

GeneratedTensorConfig
generate_tensor_config(const std::optional<std::vector<neml2::Dtype>> & gen_dtypes,
                       const std::optional<std::vector<neml2::Device>> & gen_devices)
{
  auto dtype = GENERATE_REF(from_range(gen_dtypes ? *gen_dtypes : dtypes()));
  auto device = GENERATE_REF(from_range(gen_devices ? *gen_devices : devices()));
  return {dtype, device};
}

template <typename T>
GeneratedTensorShape
generate_tensor_shape(const std::optional<std::vector<neml2::TensorShape>> & dynamic_shapes,
                      const std::optional<std::vector<neml2::TensorShape>> & intmd_shapes,
                      const std::optional<std::vector<neml2::TensorShape>> & base_shapes)
{
  auto dynamic_sizes = GENERATE_REF(from_range(dynamic_shapes ? *dynamic_shapes : shapes()));
  auto intmd_sizes = GENERATE_REF(from_range(intmd_shapes ? *intmd_shapes : shapes()));

  if constexpr (std::is_same_v<T, neml2::Tensor>)
  {
    auto base_sizes = GENERATE_REF(from_range(base_shapes ? *base_shapes : shapes()));
    return {dynamic_sizes, intmd_sizes, base_sizes};
  }
  else
  {
    if (base_shapes.has_value())
      throw neml2::NEMLException(
          "Base shapes cannot be specified when generating shapes for primitive types");
    return {dynamic_sizes, intmd_sizes, {T::const_base_sizes}};
  }
}

bool
match_tensor_config(const neml2::Tensor & tensor, const GeneratedTensorConfig & config)
{
  if (tensor.dtype() != config.dtype)
    return false;
  if (tensor.device() != config.device)
    return false;
  return true;
}

bool
match_tensor_shape(const neml2::Tensor & tensor, const GeneratedTensorShape & shape)
{
  if (tensor.dynamic_sizes() != shape.dynamic_sizes)
    return false;
  if (tensor.intmd_sizes() != neml2::TensorShapeRef(shape.intmd_sizes))
    return false;
  if (tensor.base_sizes() != neml2::TensorShapeRef(shape.base_sizes))
    return false;
  return true;
}

template <>
neml2::ATensor
generate_random_tensor(const GeneratedTensorConfig & cfg, const GeneratedTensorShape & shape)
{
  if (cfg.dtype == neml2::kFloat16 || cfg.dtype == neml2::kFloat32 || cfg.dtype == neml2::kFloat64)
    return at::rand(shape.sizes, cfg.options);
  else
    return at::randint(0, 100, shape.sizes, cfg.options);
}

template <class T>
T
generate_random_tensor(const GeneratedTensorConfig & cfg, const GeneratedTensorShape & shape)
{
  auto t = generate_random_tensor<neml2::ATensor>(cfg, shape);
  return T(t, shape.dynamic_sizes, shape.intmd_dim);
}

template <>
neml2::ATensor
generate_full_tensor(const GeneratedTensorConfig & cfg,
                     const GeneratedTensorShape & shape,
                     const neml2::CScalar & value)
{
  return at::full(shape.sizes, value, cfg.options);
}

template <class T>
T
generate_full_tensor(const GeneratedTensorConfig & cfg,
                     const GeneratedTensorShape & shape,
                     const neml2::CScalar & value)
{
  auto t = generate_full_tensor<neml2::ATensor>(cfg, shape, value);
  return T(t, shape.dynamic_sizes, shape.intmd_dim);
}

using namespace neml2;
#define INSTANTIATE(T)                                                                             \
  template GeneratedTensorShape generate_tensor_shape<T>(                                          \
      const std::optional<std::vector<neml2::TensorShape>> &,                                      \
      const std::optional<std::vector<neml2::TensorShape>> &,                                      \
      const std::optional<std::vector<neml2::TensorShape>> &);                                     \
  template T generate_random_tensor<T>(const GeneratedTensorConfig &,                              \
                                       const GeneratedTensorShape &);                              \
  template T generate_full_tensor<T>(                                                              \
      const GeneratedTensorConfig &, const GeneratedTensorShape &, const CScalar &)
FOR_ALL_TENSORBASE(INSTANTIATE);
} // namespace test
