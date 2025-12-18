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

#include "neml2/tensors/crystallography.h"

#include "neml2/tensors/Transformable.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/misc/assertions.h"

#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/functions/acos.h"
#include "neml2/tensors/functions/tr.h"
#include "neml2/tensors/functions/min.h"
#include "neml2/tensors/functions/argmin.h"

namespace neml2::crystallography
{

namespace symmetry_operators
{
Quaternion
tetragonal(const TensorOptions & options)
{
  return Quaternion::create({{o, z, z, z},
                             {z, z, o, z},
                             {z, o, z, z},
                             {z, z, z, o},
                             {a, z, z, -a},
                             {a, z, z, a},
                             {z, a, a, z},
                             {z, -a, a, z}},
                            /*intmd_dim=*/1,
                            options);
}

Quaternion
hexagonal(const TensorOptions & options)
{
  return Quaternion::create({{o, z, z, z},
                             {-h, z, z, b},
                             {h, z, z, b},
                             {b, z, z, -h},
                             {z, z, z, o},
                             {b, z, z, h},
                             {z, -h, b, z},
                             {z, o, z, z},
                             {z, h, b, z},
                             {z, b, h, z},
                             {z, z, o, z},
                             {z, b, -h, z}},
                            /*intmd_dim=*/1,
                            options);
}

Quaternion
cubic(const TensorOptions & options)
{
  return Quaternion::create({{o, z, z, z},   {h, h, h, h},    {-h, h, h, h},  {h, -h, h, h},
                             {h, h, -h, h},  {-h, -h, -h, h}, {h, -h, -h, h}, {-h, -h, h, h},
                             {-h, h, -h, h}, {z, z, o, z},    {z, z, z, o},   {z, o, z, z},
                             {z, -a, z, a},  {z, a, z, a},    {a, z, a, z},   {a, z, -a, z},
                             {z, z, -a, a},  {a, a, z, z},    {a, -a, z, z},  {z, z, a, a},
                             {z, -a, a, z},  {a, z, z, -a},   {z, a, a, z},   {a, z, z, a}},
                            /*intmd_dim=*/1,
                            options);
}
} // namespace symmetry_operators

R2
symmetry(const std::string & orbifold, const TensorOptions & options)
{
  using namespace symmetry_operators;

  if (orbifold == "432")
    return transform_from_quaternion(cubic(options));

  if (orbifold == "23")
    return transform_from_quaternion(cubic(options).intmd_index({indexing::Slice(0, 12)}));

  if (orbifold == "622")
    return transform_from_quaternion(hexagonal(options));

  if (orbifold == "32")
    return transform_from_quaternion(
        intmd_cat({hexagonal(options).intmd_index({indexing::Slice(0, 3)}),
                   hexagonal(options).intmd_index({indexing::Slice(9, 12)})}));

  if (orbifold == "6")
    return transform_from_quaternion(hexagonal(options).intmd_index({indexing::Slice(0, 6)}));

  if (orbifold == "3")
    return transform_from_quaternion(hexagonal(options).intmd_index({indexing::Slice(0, 3)}));

  if (orbifold == "42")
    return transform_from_quaternion(tetragonal(options));

  if (orbifold == "4")
    return transform_from_quaternion(
        intmd_cat({tetragonal(options).intmd_index({indexing::Slice(0, 1)}),
                   tetragonal(options).intmd_index({indexing::Slice(3, 6)})}));

  if (orbifold == "222")
    return transform_from_quaternion(tetragonal(options).intmd_index({indexing::Slice(0, 4)}));

  if (orbifold == "2")
    return transform_from_quaternion(tetragonal(options).intmd_index({indexing::Slice(0, 2)}));

  if (orbifold == "1")
    return transform_from_quaternion(tetragonal(options).intmd_index({indexing::Slice(0, 1)}));

  throw NEMLException("Unknown crystal class " + orbifold);
}

Vec
unique_bidirectional(const R2 & ops, const Vec & inp)
{
  neml_assert_dbg(inp.intmd_dim() == 0, "inp must be a single Vec without intermediate dimensions");
  // Batched tensor with all possible answers
  auto options = ops * inp;
  // I think we have to go one by one...
  // Slightly annoying that while Vec and ATensor are convertible a
  // list of Vecs aren't convertable into a TensorList
  std::vector<Vec> unique{options.intmd_index({0})};
  Vec unique_vecs = intmd_stack(unique);
  for (Size i = 1; i < options.intmd_size(0); i++)
  {
    auto vi = options.intmd_index({i});
    // Compares list of vectors to vector to figure out if any are the same
    auto same = [](const Vec & a, const Vec & b) { return at::any(at::all(at::isclose(a, b), 1)); };
    if (!(same(unique_vecs, vi).item<bool>() || same(unique_vecs, -vi).item<bool>()))
    {
      unique.emplace_back(vi);
      unique_vecs = intmd_stack(unique);
    }
  }
  // Get the batch of all possible answers
  return unique_vecs;
}

Scalar
misorientation(const Rot & r1, const Rot & r2, std::string orbifold)
{
  neml_assert_broadcastable_dbg(r1, r2);
  const auto [r1a, r2a, i] = utils::align_intmd_dim(r1, r2);

  R2 r1_mat = r1a.intmd_unsqueeze(-1).euler_rodrigues();
  R2 r2_mat = r2a.intmd_unsqueeze(-1).euler_rodrigues();
  R2 S_mat = symmetry(orbifold, r1.options());

  R2 prod = r1_mat.transpose() * S_mat * r2_mat;

  return intmd_min(acos(clamp((tr(prod) - 1.0) / 2.0, -1.0, 1.0)), -1);
}

Rot
move_to_fundamental_zone(const Rot & r, std::string orbifold, Rot ref)
{
  neml_assert_dbg(r.intmd_dim() == 0, "Input must not have intermediate dimensions");

  auto orig_dynamic_sizes = r.dynamic_sizes();
  auto flattened_r = r.batch_flatten();

  auto ops = symmetry(orbifold, r.options());
  auto poss = Rot::fill_matrix(ops * flattened_r.euler_rodrigues().intmd_unsqueeze(-1));

  auto dist = poss.dist(ref.intmd_unsqueeze(-1).to(r.options()));

  // At this point we need to go the torch tensors
  auto idx = argmin(dist, -1);
  auto batch_idx = at::arange(flattened_r.size(0)).to(idx.device());
  auto data = at::Tensor(poss).index({batch_idx, idx});

  return Rot(data, /*intmd_dim=*/0).batch_reshape(orig_dynamic_sizes, {});
}

} // namespace neml2
