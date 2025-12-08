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

#include <utility>

#include "neml2/equation_systems/HVector.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/inner.h"
#include "neml2/tensors/functions/norm_sq.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/cat.h"

namespace neml2
{

std::vector<TensorShape>
shape_refs_to_shapes(const std::vector<TensorShapeRef> & shape_refs)
{
  std::vector<TensorShape> shapes;
  shapes.reserve(shape_refs.size());
  for (const auto & sr : shape_refs)
    shapes.emplace_back(sr);
  return shapes;
}

std::vector<TensorShapeRef>
shapes_to_shape_refs(const std::vector<TensorShape> & shapes)
{
  std::vector<TensorShapeRef> shape_refs;
  shape_refs.reserve(shapes.size());
  for (const auto & s : shapes)
    shape_refs.emplace_back(s);
  return shape_refs;
}

std::vector<std::size_t>
select_subblock_indices(OptionalArrayRef<std::size_t> blocks, std::size_t n)
{
  if (blocks)
    return blocks->vec();

  std::vector<std::size_t> default_blocks;
  default_blocks.resize(n);
  std::iota(default_blocks.begin(), default_blocks.end(), 0);
  return default_blocks;
}

std::vector<TensorShapeRef>
select_shapes(const std::vector<TensorShape> & all_shapes, ArrayRef<std::size_t> blocks)
{
  neml_assert(blocks.size() <= all_shapes.size(),
              "HVector/Matrix::assemble: number of sub-blocks (",
              blocks.size(),
              ") to assemble exceeds total number of sub-blocks (",
              all_shapes.size(),
              ").");
  std::vector<TensorShapeRef> shapes(blocks.size());
  for (std::size_t i = 0; i < shapes.size(); ++i)
  {
    neml_assert(blocks[i] < all_shapes.size(),
                "HVector/Matrix::assemble: block index (",
                blocks[i],
                ") out of bounds (",
                all_shapes.size(),
                ").");
    shapes[i] = all_shapes[blocks[i]];
  }
  return shapes;
}

std::vector<Size>
numel(const std::vector<TensorShapeRef> & shapes)
{
  std::vector<Size> split_sizes(shapes.size());
  for (std::size_t i = 0; i < shapes.size(); ++i)
    split_sizes[i] = utils::numel(shapes[i]);
  return split_sizes;
}

HVector::HVector(const std::vector<TensorShapeRef> & shapes)
  : _shapes(shape_refs_to_shapes(shapes))
{
  _data.resize(_shapes.size());
}

HVector::HVector(std::vector<TensorShape> shapes)
  : _shapes(std::move(shapes))
{
  _data.resize(_shapes.size());
}

HVector::HVector(std::vector<Tensor> v, const std::vector<TensorShapeRef> & shapes)
  : HeterogeneousData(std::move(v)),
    _shapes(shape_refs_to_shapes(shapes))
{
}

HVector::HVector(std::vector<Tensor> v, std::vector<TensorShape> shapes)
  : HeterogeneousData(std::move(v)),
    _shapes(std::move(shapes))
{
}

std::vector<TensorShapeRef>
HVector::block_sizes() const
{
  return shapes_to_shape_refs(_shapes);
}

const Tensor &
HVector::operator[](std::size_t i) const
{
  neml_assert(i < n(), "HVector::operator[]: index (", i, ") out of bounds (", n(), ")");
  return _data[i];
}

Tensor &
HVector::operator[](std::size_t i)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<Tensor &>(std::as_const(*this).operator[](i));
}

HVector
HVector::operator-() const
{
  std::vector<Tensor> v(n());
  for (size_t i = 0; i < n(); i++)
  {
    const auto & vi = _data[i];
    if (vi.defined())
      v[i] = -vi;
  }
  return HVector(std::move(v), block_sizes());
}

std::pair<Tensor, std::vector<Size>>
HVector::assemble(OptionalArrayRef<std::size_t> blocks) const
{
  const auto indices = select_subblock_indices(blocks, n());
  const auto shapes = select_shapes(_shapes, indices);
  const auto split_sizes = numel(shapes);

  // If a variable is not found, tensor at that position remains undefined, and all undefined tensor
  // will later be filled with zeros.
  std::vector<Tensor> vs_f(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    const auto & v = _data[indices[i]];
    if (!v.defined())
      continue;
    const auto v_f = v.base_flatten();
    vs_f[i] = v_f;
  }

  // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with zeros.
  const auto dynamic_sizes = utils::broadcast_dynamic_sizes(vs_f);
  const auto intmd_sizes = utils::broadcast_intmd_sizes(vs_f);
  for (std::size_t i = 0; i < vs_f.size(); ++i)
    if (vs_f[i].defined())
      vs_f[i] = vs_f[i].batch_expand(dynamic_sizes, intmd_sizes);
    else
      vs_f[i] = Tensor::zeros(dynamic_sizes, intmd_sizes, split_sizes[i], options());

  return {base_cat(vs_f, -1), split_sizes};
}

void
HVector::disassemble(const Tensor & vec, OptionalArrayRef<std::size_t> blocks)
{
  const auto indices = select_subblock_indices(blocks, n());
  const auto shapes = select_shapes(_shapes, indices);
  const auto split_sizes = numel(shapes);
  const auto & D = vec.dynamic_sizes();
  const auto I = vec.intmd_dim();
  const auto vs = vec.split(split_sizes, -1);

  for (std::size_t i = 0; i < indices.size(); ++i)
    _data[indices[i]] = Tensor(vs[i], D, I).base_reshape(shapes[i]);
}

void
HVector::update_data(const HVector & other)
{
  neml_assert_dbg(
      n() == other.n(), "HVector::update_data: size mismatch (", n(), ") vs (", other.n(), ")");
  for (std::size_t i = 0; i < n(); i++)
  {
    auto & v = _data[i];
    v = v.variable_data() + other[i];
  }
}

void
HVector::update(const HVector & other)
{
  neml_assert_dbg(
      n() == other.n(), "HVector::update_data: size mismatch (", n(), ") vs (", other.n(), ")");
  for (size_t i = 0; i < n(); i++)
  {
    auto & v = (*this)[i];
    v = v + other[i];
  }
}

HVector
operator+(const HVector & a, const HVector & b)
{
  neml_assert_dbg(
      a.n() == b.n(), "HVector::operator+: size mismatch (", a.n(), ") vs (", b.n(), ")");
  std::vector<Tensor> v(a.n());
  for (std::size_t i = 0; i < a.n(); i++)
  {
    const auto & ai = a[i];
    const auto & bi = b[i];
    if (ai.defined() && bi.defined())
      v[i] = ai + bi;
    else if (ai.defined())
      v[i] = ai;
    else if (bi.defined())
      v[i] = bi;
  }
  return HVector(std::move(v), a.block_sizes());
}

HVector
operator+(const Scalar & a, const HVector & b)
{
  std::vector<Tensor> v(b.n());
  for (std::size_t i = 0; i < b.n(); i++)
  {
    const auto & bi = b[i];
    v[i] = bi.defined() ? a + bi : Tensor(a);
  }
  return HVector(std::move(v), b.block_sizes());
}

HVector
operator+(const HVector & a, const Scalar & b)
{
  return b + a;
}

HVector
operator+(const CScalar & a, const HVector & b)
{
  std::vector<Tensor> v(b.n());
  for (std::size_t i = 0; i < b.n(); i++)
  {
    const auto & bi = b[i];
    v[i] = bi.defined() ? a + bi : Tensor(Scalar(a, bi.options()));
  }
  return HVector(std::move(v), b.block_sizes());
}

HVector
operator+(const HVector & a, const CScalar & b)
{
  return b + a;
}

HVector
operator-(const HVector & a, const HVector & b)
{
  neml_assert_dbg(
      a.n() == b.n(), "HVector::operator-: size mismatch (", a.n(), ") vs (", b.n(), ")");
  std::vector<Tensor> v(a.n());
  for (std::size_t i = 0; i < a.n(); i++)
  {
    const auto & ai = a[i];
    const auto & bi = b[i];
    if (ai.defined() && bi.defined())
      v[i] = ai - bi;
    else if (ai.defined())
      v[i] = ai;
    else if (bi.defined())
      v[i] = -bi;
  }
  return HVector(std::move(v), a.block_sizes());
}

HVector
operator-(const Scalar & a, const HVector & b)
{
  std::vector<Tensor> v(b.n());
  for (std::size_t i = 0; i < b.n(); i++)
  {
    const auto & bi = b[i];
    v[i] = bi.defined() ? a - bi : Tensor(a);
  }
  return HVector(std::move(v), b.block_sizes());
}

HVector
operator-(const HVector & a, const Scalar & b)
{
  return (-b) + a;
}

HVector
operator-(const CScalar & a, const HVector & b)
{
  std::vector<Tensor> v(b.n());
  for (std::size_t i = 0; i < b.n(); i++)
  {
    const auto & bi = b[i];
    v[i] = bi.defined() ? a - bi : Tensor(Scalar(a, bi.options()));
  }
  return HVector(std::move(v), b.block_sizes());
}

HVector
operator-(const HVector & a, const CScalar & b)
{
  return (-b) + a;
}

Scalar
operator*(const HVector & a, const HVector & b)
{
  neml_assert_dbg(
      a.n() == b.n(), "HVector::operator*: size mismatch (", a.n(), ") vs (", b.n(), ")");
  auto v = Scalar::zeros(a.options());
  for (std::size_t i = 0; i < a.n(); i++)
  {
    const auto & ai = a[i];
    const auto & bi = b[i];
    if (ai.defined() && bi.defined())
      v = v + inner(ai, bi);
  }
  return v;
}

HVector
operator*(const Scalar & a, const HVector & b)
{
  std::vector<Tensor> v(b.n());
  for (std::size_t i = 0; i < b.n(); i++)
  {
    const auto & bi = b[i];
    if (bi.defined())
      v[i] = a * bi;
  }
  return HVector(std::move(v), b.block_sizes());
}

HVector
operator*(const HVector & a, const Scalar & b)
{
  return b * a;
}

HVector
operator*(const CScalar & a, const HVector & b)
{
  std::vector<Tensor> v(b.n());
  for (std::size_t i = 0; i < b.n(); i++)
  {
    const auto & bi = b[i];
    if (bi.defined())
      v[i] = a * bi;
  }
  return HVector(std::move(v), b.block_sizes());
}

HVector
operator*(const HVector & a, const CScalar & b)
{
  return b * a;
}

HVector
operator/(const HVector & a, const Scalar & b)
{
  std::vector<Tensor> v(a.n());
  for (std::size_t i = 0; i < a.n(); i++)
  {
    const auto & ai = a[i];
    if (ai.defined())
      v[i] = ai / b;
  }
  return HVector(std::move(v), a.block_sizes());
}

HVector
operator/(const HVector & a, const CScalar & b)
{
  std::vector<Tensor> v(a.n());
  for (std::size_t i = 0; i < a.n(); i++)
  {
    const auto & ai = a[i];
    if (ai.defined())
      v[i] = ai / b;
  }
  return HVector(std::move(v), a.block_sizes());
}

Scalar
norm_sq(const HVector & v)
{
  Scalar sum_sq = Scalar::zeros();
  for (const auto & vi : v)
    if (vi.defined())
      sum_sq = sum_sq + neml2::norm_sq(vi);
  return sum_sq;
}

Scalar
norm(const HVector & v)
{
  return neml2::sqrt(norm_sq(v));
}

} // namespace neml2
