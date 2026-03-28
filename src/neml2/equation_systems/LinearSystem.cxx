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

namespace neml2
{
void
LinearSystem::init()
{
  // Note: These data structures we are setting up here serve the same purpose as LabeledAxis, and
  // yes, we are storing redundant information. This is because we are transitioning away from
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
LinearSystem::A()
{
  SparseMatrix A(_blayout->view(), _ulayout->view());
  pre_assemble(true, false, false);
  assemble(&A, nullptr, nullptr);
  post_assemble(true, false, false);
  return A;
}

SparseVector
LinearSystem::b()
{
  SparseVector b(_blayout->view());
  pre_assemble(false, false, true);
  assemble(nullptr, nullptr, &b);
  post_assemble(false, false, true);
  return b;
}

std::tuple<SparseMatrix, SparseVector>
LinearSystem::A_and_b()
{
  SparseMatrix A(_blayout->view(), _ulayout->view());
  SparseVector b(_blayout->view());
  pre_assemble(true, false, true);
  assemble(&A, nullptr, &b);
  post_assemble(true, false, true);
  return {A, b};
}

std::tuple<SparseMatrix, SparseMatrix>
LinearSystem::A_and_B()
{
  SparseMatrix A(_blayout->view(), _ulayout->view());
  SparseMatrix B(_blayout->view(), _glayout->view());
  pre_assemble(true, true, false);
  assemble(&A, &B, nullptr);
  post_assemble(true, true, false);
  return {A, B};
}

std::tuple<SparseMatrix, SparseMatrix, SparseVector>
LinearSystem::A_and_B_and_b()
{
  SparseMatrix A(_blayout->view(), _ulayout->view());
  SparseMatrix B(_blayout->view(), _glayout->view());
  SparseVector b(_blayout->view());
  pre_assemble(true, true, true);
  assemble(&A, &B, &b);
  post_assemble(true, true, true);
  return {A, B, b};
}

const std::shared_ptr<AxisLayout> &
LinearSystem::ulayout() const
{
  return _ulayout;
}

const std::shared_ptr<AxisLayout> &
LinearSystem::glayout() const
{
  return _glayout;
}

const std::shared_ptr<AxisLayout> &
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
}

} // namespace neml2
