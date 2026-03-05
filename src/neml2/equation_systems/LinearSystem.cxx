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

#include "neml2/equation_systems/LinearSystem.h"
#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/misc/errors.h"
#include "neml2/equation_systems/SparseTensorList.h"

namespace neml2
{
void
LinearSystem::init()
{
  // Note: These data structures we are setting here serve the same purpose as LabeledAxis, and yes,
  // we are storing redundant information. This is because we are transitioning away from
  // LabeledAxis. In future versions, we will remove LabeledAxis and only use these data structures.
  //
  // Also note: only models that are wrapped as equation systems need to define these maps. Regular
  // feed-forward models do not need to define these maps. This is an important distinction from
  // LabeledAxis which is always defined for any model.
  //
  // Another note: Right now we can "smartly" determine these maps based on variable subaxes. In the
  // future, since we are removing LabeledAxis, we will need let the user explicitly define these
  // maps, i.e., from within the input files.
  _ulayout = setup_ulayout();
  _glayout = setup_glayout();
  _blayout = setup_blayout();
}

void
LinearSystem::u_changed()
{
}

void
LinearSystem::g_changed()
{
  _A_up_to_date = false;
  _B_up_to_date = false;
  _b_up_to_date = false;
}

SparseMatrix
LinearSystem::A(std::size_t bgroup_idx, std::size_t ugroup_idx)
{
  SparseTensorList A;
  pre_assemble(true, false, false);
  assemble(&A, nullptr, nullptr, bgroup_idx, ugroup_idx);
  post_assemble(true, false, false);
  return {A, _blayout[bgroup_idx], _ulayout[ugroup_idx]};
}

SparseVector
LinearSystem::b(std::size_t group_idx)
{
  SparseTensorList b;
  pre_assemble(false, false, true);
  assemble(nullptr, nullptr, &b, group_idx);
  post_assemble(false, false, true);
  return {b, _blayout[group_idx]};
}

std::tuple<SparseMatrix, SparseVector>
LinearSystem::A_and_b(std::size_t bgroup_idx, std::size_t ugroup_idx)
{
  SparseTensorList A, b;
  pre_assemble(true, false, true);
  assemble(&A, nullptr, &b, bgroup_idx, ugroup_idx);
  post_assemble(true, false, true);
  return {SparseMatrix(A, _blayout[bgroup_idx], _ulayout[ugroup_idx]),
          SparseVector(b, _blayout[bgroup_idx])};
}

std::tuple<SparseMatrix, SparseMatrix>
LinearSystem::A_and_B(std::size_t bgroup_idx, std::size_t ugroup_idx)
{
  SparseTensorList A, B;
  pre_assemble(true, true, false);
  assemble(&A, &B, nullptr, bgroup_idx, ugroup_idx);
  post_assemble(true, true, false);
  return {SparseMatrix(A, _blayout[bgroup_idx], _ulayout[ugroup_idx]),
          SparseMatrix(B, _blayout[bgroup_idx], _ulayout[ugroup_idx])};
}

std::tuple<SparseMatrix, SparseMatrix, SparseVector>
LinearSystem::A_and_B_and_b(std::size_t bgroup_idx, std::size_t ugroup_idx)
{
  SparseTensorList A, B, b;
  pre_assemble(true, true, true);
  assemble(&A, &B, &b, bgroup_idx, ugroup_idx);
  post_assemble(true, true, true);
  return {SparseMatrix(A, _blayout[bgroup_idx], _ulayout[ugroup_idx]),
          SparseMatrix(B, _blayout[bgroup_idx], _ulayout[ugroup_idx]),
          SparseVector(b, _blayout[bgroup_idx])};
}

static const std::shared_ptr<AxisLayout> &
resolve_group(std::size_t idx,
              const std::vector<std::shared_ptr<AxisLayout>> & data,
              const std::string & name)
{
  if (data.empty())
    throw NEMLException("No groups are defined for '" + name + "'.");
  if (idx >= data.size())
    throw NEMLException("Invalid group index " + std::to_string(idx) + " for " + name +
                        ". Available group indices are 0.." + std::to_string(data.size() - 1) +
                        ".");
  return data[idx];
}

const std::shared_ptr<AxisLayout> &
LinearSystem::ulayout(std::size_t group_idx) const
{
  return resolve_group(group_idx, _ulayout, "unknown variables");
}

const std::shared_ptr<AxisLayout> &
LinearSystem::glayout() const
{
  return _glayout;
}

const std::shared_ptr<AxisLayout> &
LinearSystem::blayout(std::size_t group_idx) const
{
  return resolve_group(group_idx, _blayout, "RHS variables");
}

void
LinearSystem::pre_assemble(bool /*A*/, bool /*B*/, bool /*b*/)
{
}

void
LinearSystem::post_assemble(bool A, bool B, bool b)
{
  _A_up_to_date |= A;
  _B_up_to_date |= B;
  _b_up_to_date |= A || B || b;
}

} // namespace neml2
