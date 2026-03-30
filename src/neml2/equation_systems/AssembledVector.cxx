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

#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/misc/assertions.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/misc/defaults.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/inner.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{

AssembledVector::AssembledVector(AxisLayout l)
  : tensors(l.ngroup()),
    layout(std::move(l))
{
}

AssembledVector::AssembledVector(AxisLayout l, std::vector<Tensor> t)
  : tensors(std::move(t)),
    layout(std::move(l))
{
  neml_assert_dbg(tensors.size() == layout.ngroup(),
                  "Number of tensors must match the layout group size");
}

TensorOptions
AssembledVector::options() const
{
  for (const auto & t : tensors)
    if (t.defined())
      return t.options();
  return default_tensor_options();
}

AssembledVector
AssembledVector::group(std::size_t i) const
{
  neml_assert(i < layout.ngroup(), "Group index out of range");
  return AssembledVector(layout.group(i), {tensors[i]});
}

SparseVector
AssembledVector::disassemble() const
{
  std::vector<Tensor> sp_tensors(layout.nvar());

  for (std::size_t grp = 0; grp < layout.ngroup(); ++grp)
  {
    const auto & t = tensors[grp];
    const auto [istart, iend] = layout.group_offsets(grp);
    const auto istr = layout.istr(grp);
    const bool assemble_intmd = (istr == AxisLayout::IStructure::DENSE);

    neml_assert_dbg(
        t.base_dim() == 1, "disassemble expects base dimension of 1, got ", t.base_dim());
    if (assemble_intmd)
      neml_assert_dbg(t.intmd_dim() == 0,
                      "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                      t.intmd_dim());

    const auto ss = layout.group(grp).storage_sizes(assemble_intmd);
    const auto & D = t.dynamic_sizes();
    const auto I = t.intmd_dim();
    const auto vs = t.split(ss, -1);

    for (std::size_t i = istart; i < iend; ++i)
    {
      auto ti = Tensor(vs[i - istart], D, I);
      if (!assemble_intmd)
        sp_tensors[i] = ti.base_reshape(layout.base_sizes(i));
      else
        sp_tensors[i] = from_assembly<1>(ti, {layout.intmd_sizes(i)}, {layout.base_sizes(i)});
    }
  }

  return SparseVector(layout, std::move(sp_tensors));
}

AssembledVector
operator-(const AssembledVector & v)
{
  auto r = v;
  for (auto & t : r.tensors)
    if (t.defined())
      t = -t;
  return r;
}

AssembledVector
operator+(const AssembledVector & a, const AssembledVector & b)
{
  std::vector<Tensor> tensors(a.tensors.size());
  for (std::size_t i = 0; i < a.tensors.size(); ++i)
  {
    if (a.tensors[i].defined() && b.tensors[i].defined())
      tensors[i] = a.tensors[i] + b.tensors[i];
    else if (a.tensors[i].defined())
      tensors[i] = a.tensors[i];
    else if (b.tensors[i].defined())
      tensors[i] = b.tensors[i];
  }
  return AssembledVector(a.layout, std::move(tensors));
}

AssembledVector
operator-(const AssembledVector & a, const AssembledVector & b)
{
  std::vector<Tensor> tensors(a.tensors.size());
  for (std::size_t i = 0; i < a.tensors.size(); ++i)
  {
    if (a.tensors[i].defined() && b.tensors[i].defined())
      tensors[i] = a.tensors[i] - b.tensors[i];
    else if (a.tensors[i].defined())
      tensors[i] = a.tensors[i];
    else if (b.tensors[i].defined())
      tensors[i] = -b.tensors[i];
  }
  return AssembledVector(a.layout, std::move(tensors));
}

AssembledVector
operator*(const Scalar & s, const AssembledVector & v)
{
  std::vector<Tensor> tensors(v.tensors.size());
  for (std::size_t i = 0; i < v.tensors.size(); ++i)
    if (v.tensors[i].defined())
      tensors[i] = s * v.tensors[i];
  return AssembledVector(v.layout, std::move(tensors));
}

AssembledVector
operator*(const AssembledVector & v, const Scalar & s)
{
  return s * v;
}

Scalar
operator*(const AssembledVector & a, const AssembledVector & b)
{
  neml_assert(a.layout.ngroup() == b.layout.ngroup(),
              "Inner product requires vectors with the same number of groups, got ",
              a.layout.ngroup(),
              " and ",
              b.layout.ngroup());
  auto result = Scalar::zeros(a.options());
  for (std::size_t i = 0; i < a.layout.ngroup(); ++i)
    if (a.tensors[i].defined() && b.tensors[i].defined())
      result = result + neml2::inner(a.tensors[i], b.tensors[i]);
  return result;
}

Scalar
norm_sq(const AssembledVector & v)
{
  return v * v;
}

Scalar
norm(const AssembledVector & v)
{
  return neml2::sqrt(norm_sq(v));
}

} // namespace neml2
