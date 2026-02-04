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
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/base/LabeledAxisAccessor.h"

namespace neml2
{

void
LinearSystem::setup()
{
  EquationSystem::setup();

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
  return _bmap.size();
}

std::size_t
LinearSystem::n() const
{
  return _umap.size();
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
LinearSystem::umap() const
{
  return _umap;
}

const std::vector<TensorShape> &
LinearSystem::intmd_ulayout() const
{
  neml_assert(_intmd_ulayout.has_value(),
              "Intermediate shapes for unknowns requested but not set up.");
  return _intmd_ulayout.value();
}

const std::vector<TensorShape> &
LinearSystem::ulayout() const
{
  return _ulayout;
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
LinearSystem::bmap() const
{
  return _bmap;
}

const std::vector<TensorShape> &
LinearSystem::intmd_blayout() const
{
  neml_assert(_intmd_blayout.has_value(), "Intermediate shapes for RHS requested but not set up.");
  return _intmd_blayout.value();
}

const std::vector<TensorShape> &
LinearSystem::blayout() const
{
  return _blayout;
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
