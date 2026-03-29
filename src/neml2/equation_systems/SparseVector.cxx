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

#include "neml2/equation_systems/SparseVector.h"
#include "neml2/misc/assertions.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/misc/defaults.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/functions/inner.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{

SparseVector::SparseVector(const AxisLayout & l)
  : tensors(l.size()),
    layout(l)
{
}

SparseVector::SparseVector(const AxisLayout & l, std::vector<Tensor> t)
  : tensors(std::move(t)),
    layout(l)
{
  neml_assert_dbg(tensors.size() == layout.size(), "Number of tensors must match the layout size");
}

TensorOptions
SparseVector::options() const
{
  for (const auto & t : tensors)
    if (t.defined())
      return t.options();
  return default_tensor_options();
}

std::size_t
SparseVector::ngroup() const
{
  return layout.ngroup();
}

SparseVector
SparseVector::group(std::size_t i) const
{
  auto [start, end] = layout.group_offsets(i);
  std::vector<Tensor> ts(tensors.begin() + Size(start), tensors.begin() + Size(end));
  return SparseVector(layout.group(i), std::move(ts));
}

std::size_t
SparseVector::size() const
{
  return tensors.size();
}

Tensor
SparseVector::assemble(bool assemble_intmd) const
{
  // convert to assembly format
  std::vector<Tensor> tf(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i)
  {
    const auto & ti = tensors[i];
    if (!ti.defined())
      continue;
    if (!assemble_intmd)
      tf[i] = ti.base_flatten();
    else
      tf[i] = to_assembly<1>(ti, {layout.intmd_sizes(i)}, {layout.base_sizes(i)});
  }

  // determine tensor options
  auto opt = default_tensor_options();
  for (const auto & t : tf)
    if (t.defined())
    {
      opt = t.options();
      break;
    }

  // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with zeros.
  const auto new_dynamic_sizes = utils::broadcast_dynamic_sizes(tf);
  const auto new_intmd_sizes = utils::broadcast_intmd_sizes(tf);
  for (std::size_t i = 0; i < tf.size(); ++i)
  {
    auto & tfi = tf[i];
    if (tfi.defined())
      tfi = tfi.batch_expand(new_dynamic_sizes, new_intmd_sizes);
    else
    {
      auto s = utils::numel(layout.base_sizes(i));
      if (assemble_intmd)
        s *= utils::numel(layout.intmd_sizes(i));
      tfi = Tensor::zeros(new_dynamic_sizes, new_intmd_sizes, s, opt);
    }
  }

  return base_cat(tf, -1);
}

void
SparseVector::disassemble(const Tensor & t, bool assemble_intmd)
{
  neml_assert_dbg(t.base_dim() == 1, "disassemble expects base dimension of 1, got ", t.base_dim());
  if (assemble_intmd)
    neml_assert_dbg(t.intmd_dim() == 0,
                    "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                    t.intmd_dim());

  const auto ss = layout.storage_sizes(assemble_intmd);
  const auto & D = t.dynamic_sizes();
  const auto I = t.intmd_dim();
  const auto vs = t.split(ss, -1);
  const auto n = vs.size();

  for (std::size_t i = 0; i < n; ++i)
  {
    auto ti = Tensor(vs[i], D, I);
    if (!assemble_intmd)
      tensors[i] = ti.base_reshape(layout.base_sizes(i));
    else
      tensors[i] = from_assembly<1>(ti, {layout.intmd_sizes(i)}, {layout.base_sizes(i)});
  }
}

SparseVector
operator-(const SparseVector & a)
{
  std::vector<Tensor> t(a.size());
  for (std::size_t i = 0; i < t.size(); i++)
    if (a.tensors[i].defined())
      t[i] = -a.tensors[i];
  return SparseVector(a.layout, std::move(t));
}

SparseVector
operator-(const SparseVector & a, const SparseVector & b)
{
  return a + (-b);
}

SparseVector
operator+(const SparseVector & a, const SparseVector & b)
{
  neml_assert(a.size() == b.size(),
              "Incompatible sizes in SparseVector addition, got ",
              a.size(),
              " and ",
              b.size());
  std::vector<Tensor> t(a.size());
  for (std::size_t i = 0; i < t.size(); i++)
    if (a.tensors[i].defined() && b.tensors[i].defined())
      t[i] = a.tensors[i] + b.tensors[i];
    else if (a.tensors[i].defined())
      t[i] = a.tensors[i];
    else if (b.tensors[i].defined())
      t[i] = b.tensors[i];
  return SparseVector(a.layout, std::move(t));
}

SparseVector
operator*(const Scalar & s, const SparseVector & a)
{
  std::vector<Tensor> t(a.size());
  for (std::size_t i = 0; i < a.size(); i++)
    if (a.tensors[i].defined())
      t[i] = s * a.tensors[i];
  return SparseVector(a.layout, std::move(t));
}

SparseVector
operator*(const SparseVector & a, const Scalar & s)
{
  return s * a;
}

Scalar
operator*(const SparseVector & a, const SparseVector & b)
{
  neml_assert(a.size() == b.size(),
              "Incompatible sizes in SparseVector inner product, got ",
              a.size(),
              " and ",
              b.size());
  Scalar s;
  for (std::size_t i = 0; i < a.size(); i++)
    if (a.tensors[i].defined() && b.tensors[i].defined())
    {
      auto ab = neml2::inner(a.tensors[i], b.tensors[i]);
      if (!s.defined())
        s = ab;
      else
        s = s + ab;
    }
  if (s.defined())
    return s;
  return Scalar::zeros();
}

Scalar
norm_sq(const SparseVector & a)
{
  return a * a;
}

Scalar
norm(const SparseVector & a)
{
  return neml2::sqrt(norm_sq(a));
}

} // namespace neml2
