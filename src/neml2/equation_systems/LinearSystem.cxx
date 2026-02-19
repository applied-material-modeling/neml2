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
#include "neml2/misc/assertions.h"
#include "neml2/misc/errors.h"
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/base/LabeledAxisAccessor.h"

namespace neml2
{
template <typename T>
const std::vector<T> &
LinearSystem::resolve_group(std::size_t group_idx,
                            const std::vector<std::vector<T>> & group_data,
                            const std::string & object_name)
{
  if (group_data.empty())
    throw NEMLException("No groups are defined for '" + object_name + "'.");

  if (group_idx >= group_data.size())
    throw NEMLException("Invalid group index " + std::to_string(group_idx) + " for '" +
                        object_name + "'. Available group indices are 0.." +
                        std::to_string(group_data.size() - 1) + ".");

  return group_data[group_idx];
}

template <typename T>
std::vector<T>
LinearSystem::flatten_groups(const std::vector<std::vector<T>> & group_data)
{
  std::vector<T> flattened;
  for (const auto & group : group_data)
    for (const auto & item : group)
      flattened.push_back(item);
  return flattened;
}

void
LinearSystem::init()
{
  // Note: These data structures we are setting here serve the same purpose as LabeledAxis, and yes,
  // we are storing redundant information. This is because we are transitioning away from
  // LabeledAxis. In future versions, we will remove LabeledAxis and only use these data structures.
  //
  // Also note: only nonlinear systems need to define these maps. Regular feed-forward models do not
  // need to define these maps. This is an important distinction from LabeledAxis which is always
  // defined.
  //
  // Another note: Right now we can "smartly" determine these maps based on variable subaxes. In the
  // future, since we are removing LabeledAxis, we will need let the user explicitly define these
  // maps, i.e., from within the input files.

  _umap = setup_umap();
  _ulayout = setup_ulayout();

  _gmap = setup_gmap();
  _glayout = setup_glayout();

  _bmap = setup_bmap();
  _blayout = setup_blayout();
}

std::size_t
LinearSystem::m() const
{
  std::size_t total = 0;
  for (const auto & group : _bmap)
    total += group.size();
  return total;
}

std::size_t
LinearSystem::n() const
{
  std::size_t total = 0;
  for (const auto & group : _umap)
    total += group.size();
  return total;
}

std::size_t
LinearSystem::p() const
{
  return _gmap.size();
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

SparseTensorList
LinearSystem::A()
{
  SparseTensorList A;
  pre_assemble(true, false, false);
  assemble(&A, nullptr, nullptr);
  post_assemble(true, false, false);
  return A;
}

SparseTensorList
LinearSystem::b()
{
  SparseTensorList b;
  pre_assemble(false, false, true);
  assemble(nullptr, nullptr, &b);
  post_assemble(false, false, true);
  return b;
}

std::tuple<SparseTensorList, SparseTensorList>
LinearSystem::A_and_b()
{
  SparseTensorList A, b;
  pre_assemble(true, false, true);
  assemble(&A, nullptr, &b);
  post_assemble(true, false, true);
  return {A, b};
}

std::tuple<SparseTensorList, SparseTensorList>
LinearSystem::A_and_B()
{
  SparseTensorList A, B;
  pre_assemble(true, true, false);
  assemble(&A, &B, nullptr);
  post_assemble(true, true, false);
  return {A, B};
}

std::tuple<SparseTensorList, SparseTensorList, SparseTensorList>
LinearSystem::A_and_B_and_b()
{
  SparseTensorList A, B, b;
  pre_assemble(true, true, true);
  assemble(&A, &B, &b);
  post_assemble(true, true, true);
  return {A, B, b};
}

const std::vector<LabeledAxisAccessor> &
LinearSystem::umap(std::size_t group_idx) const
{
  return resolve_group(group_idx, _umap, "umap");
}

std::vector<LabeledAxisAccessor>
LinearSystem::full_umap() const
{
  return flatten_groups(_umap);
}

const std::vector<TensorShape> &
LinearSystem::intmd_ulayout(std::size_t group_idx) const
{
  neml_assert(_intmd_ulayout.has_value(),
              "Intermediate shapes for unknowns requested but not set up.");
  return resolve_group(group_idx, _intmd_ulayout.value(), "intmd_ulayout");
}

const std::vector<TensorShape> &
LinearSystem::ulayout(std::size_t group_idx) const
{
  return resolve_group(group_idx, _ulayout, "ulayout");
}

const std::vector<LabeledAxisAccessor> &
LinearSystem::gmap() const
{
  return _gmap;
}

const std::vector<TensorShape> &
LinearSystem::intmd_glayout() const
{
  neml_assert(_intmd_glayout.has_value(),
              "Intermediate shapes for given variables requested but not set up.");
  return _intmd_glayout.value();
}

const std::vector<TensorShape> &
LinearSystem::glayout() const
{
  return _glayout;
}

const std::vector<LabeledAxisAccessor> &
LinearSystem::bmap(std::size_t group_idx) const
{
  return resolve_group(group_idx, _bmap, "bmap");
}

std::vector<LabeledAxisAccessor>
LinearSystem::full_bmap() const
{
  return flatten_groups(_bmap);
}

const std::vector<TensorShape> &
LinearSystem::intmd_blayout(std::size_t group_idx) const
{
  neml_assert(_intmd_blayout.has_value(), "Intermediate shapes for RHS requested but not set up.");
  return resolve_group(group_idx, _intmd_blayout.value(), "intmd_blayout");
}

const std::vector<TensorShape> &
LinearSystem::blayout(std::size_t group_idx) const
{
  return resolve_group(group_idx, _blayout, "blayout");
}

std::size_t
LinearSystem::n_ugroup() const
{
  return _umap.size();
}

std::size_t
LinearSystem::n_bgroup() const
{
  return _bmap.size();
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

  if (!_intmd_ulayout.has_value())
    _intmd_ulayout = setup_intmd_ulayout();

  if (!_intmd_glayout.has_value())
    _intmd_glayout = setup_intmd_glayout();

  if (!_intmd_blayout.has_value())
    _intmd_blayout = setup_intmd_blayout();
}

} // namespace neml2
