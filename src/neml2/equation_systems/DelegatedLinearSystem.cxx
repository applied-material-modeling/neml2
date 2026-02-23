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

#include "neml2/equation_systems/DelegatedLinearSystem.h"

#include "neml2/misc/assertions.h"
#include "neml2/misc/errors.h"

namespace neml2
{
DelegatedLinearSystem::DelegatedLinearSystem(const Layout & u_layout,
                                             const Layout & g_layout,
                                             const Layout & b_layout,
                                             const SparseTensorList & A,
                                             const std::optional<SparseTensorList> & B,
                                             const std::optional<SparseTensorList> & b)
  : _u_layout(u_layout),
    _g_layout(g_layout),
    _b_layout(b_layout),
    _A(A),
    _B(B),
    _b(b),
    _u(u_layout.map.size()),
    _g(g_layout.map.size())
{
  validate_layout(_u_layout, "unknown");
  validate_layout(_g_layout, "given");
  validate_layout(_b_layout, "RHS");

  validate_size(_A, _b_layout.map.size() * _u_layout.map.size(), "A");
  if (_B)
    validate_size(*_B, _b_layout.map.size() * _g_layout.map.size(), "B");
  if (_b)
    validate_size(*_b, _b_layout.map.size(), "b");

  init();
}

void
DelegatedLinearSystem::set_operator(const SparseTensorList & A)
{
  validate_size(A, _b_layout.map.size() * _u_layout.map.size(), "A");
  _A = A;
  _A_up_to_date = false;
}

void
DelegatedLinearSystem::set_auxiliary_operator(const std::optional<SparseTensorList> & B)
{
  if (B)
    validate_size(*B, _b_layout.map.size() * _g_layout.map.size(), "B");
  _B = B;
  _B_up_to_date = false;
}

void
DelegatedLinearSystem::set_rhs(const std::optional<SparseTensorList> & b)
{
  if (b)
    validate_size(*b, _b_layout.map.size(), "b");
  _b = b;
  _b_up_to_date = false;
}

void
DelegatedLinearSystem::set_u(const SparseTensorList & u, std::size_t group_idx)
{
  validate_group_idx(group_idx, "unknown");
  validate_size(u, _u_layout.map.size(), "u");
  _u = u;
  u_changed();
}

void
DelegatedLinearSystem::set_g(const SparseTensorList & g)
{
  validate_size(g, _g_layout.map.size(), "g");
  _g = g;
  g_changed();
}

SparseTensorList
DelegatedLinearSystem::u(std::size_t group_idx) const
{
  validate_group_idx(group_idx, "unknown");
  return _u;
}

SparseTensorList
DelegatedLinearSystem::g() const
{
  return _g;
}

std::vector<std::vector<LabeledAxisAccessor>>
DelegatedLinearSystem::setup_umap()
{
  return {_u_layout.map};
}

std::vector<std::vector<TensorShape>>
DelegatedLinearSystem::setup_intmd_ulayout()
{
  return {_u_layout.intmd};
}

std::vector<std::vector<TensorShape>>
DelegatedLinearSystem::setup_ulayout()
{
  return {_u_layout.base};
}

std::vector<LabeledAxisAccessor>
DelegatedLinearSystem::setup_gmap()
{
  return _g_layout.map;
}

std::vector<TensorShape>
DelegatedLinearSystem::setup_intmd_glayout()
{
  return _g_layout.intmd;
}

std::vector<TensorShape>
DelegatedLinearSystem::setup_glayout()
{
  return _g_layout.base;
}

std::vector<std::vector<LabeledAxisAccessor>>
DelegatedLinearSystem::setup_bmap()
{
  return {_b_layout.map};
}

std::vector<std::vector<TensorShape>>
DelegatedLinearSystem::setup_intmd_blayout()
{
  return {_b_layout.intmd};
}

std::vector<std::vector<TensorShape>>
DelegatedLinearSystem::setup_blayout()
{
  return {_b_layout.base};
}

void
DelegatedLinearSystem::assemble(SparseTensorList * A,
                                SparseTensorList * B,
                                SparseTensorList * b,
                                std::size_t bgroup_idx,
                                std::size_t ugroup_idx)
{
  validate_group_idx(bgroup_idx, "RHS");
  validate_group_idx(ugroup_idx, "unknown");

  if (A)
    *A = _A;

  if (B)
  {
    if (!_B)
      throw NEMLException("Auxiliary operator B was requested but not set in DelegatedLinearSystem.");
    *B = *_B;
  }

  if (b)
  {
    if (!_b)
      throw NEMLException("Right-hand side b was requested but not set in DelegatedLinearSystem.");
    *b = *_b;
  }
}

void
DelegatedLinearSystem::validate_group_idx(std::size_t idx, const char * name)
{
  if (idx != 0)
    throw NEMLException("DelegatedLinearSystem only supports group index 0 for " + std::string(name) +
                        " variables.");
}

void
DelegatedLinearSystem::validate_size(const SparseTensorList & values,
                                     std::size_t expected,
                                     const char * name)
{
  neml_assert(values.size() == expected,
              "DelegatedLinearSystem expected ",
              name,
              " to contain ",
              expected,
              " entries, got ",
              values.size(),
              ".");
}

void
DelegatedLinearSystem::validate_layout(const Layout & layout, const char * name)
{
  if (layout.map.size() != layout.intmd.size() || layout.map.size() != layout.base.size())
    throw NEMLException("DelegatedLinearSystem " + std::string(name) +
                        " layout is inconsistent: map/intmd/base must have the same size.");
}
} // namespace neml2
