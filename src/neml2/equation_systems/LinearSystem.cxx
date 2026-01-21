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
#include "neml2/equation_systems/HVector.h"
#include "neml2/equation_systems/HMatrix.h"

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

  setup_umap_and_layout();
  setup_unmap_and_layout();
  setup_gmap_and_layout();
  setup_gnmap_and_layout();
  setup_bmap_and_layout();
  setup_current_to_old_maps();
  setup_old_to_current_maps();
}

const std::vector<LabeledAxisAccessor> &
LinearSystem::umap() const
{
  return _umap;
}

const std::vector<TensorShape> &
LinearSystem::ulayout() const
{
  return _ulayout;
}

const std::vector<LabeledAxisAccessor> &
LinearSystem::unmap() const
{
  return _unmap;
}

const std::vector<TensorShape> &
LinearSystem::unlayout() const
{
  return _unlayout;
}

const std::vector<LabeledAxisAccessor> &
LinearSystem::gmap() const
{
  return _gmap;
}

const std::vector<TensorShape> &
LinearSystem::glayout() const
{
  return _glayout;
}

const std::vector<LabeledAxisAccessor> &
LinearSystem::gnmap() const
{
  return _gnmap;
}

const std::vector<TensorShape> &
LinearSystem::gnlayout() const
{
  return _gnlayout;
}

const std::vector<LabeledAxisAccessor> &
LinearSystem::bmap() const
{
  return _bmap;
}

const std::vector<TensorShape> &
LinearSystem::blayout() const
{
  return _blayout;
}

HVector
LinearSystem::u() const
{
  return create_uvec();
}

HVector
LinearSystem::un() const
{
  return create_unvec();
}

HVector
LinearSystem::g() const
{
  return create_gvec();
}

HVector
LinearSystem::gn() const
{
  return create_gnvec();
}

HVector
LinearSystem::create_uvec() const
{
  return HVector(_ulayout);
}

HVector
LinearSystem::create_unvec() const
{
  return HVector(_unlayout);
}

HVector
LinearSystem::create_gvec() const
{
  return HVector(_glayout);
}

HVector
LinearSystem::create_gnvec() const
{
  return HVector(_gnlayout);
}

HVector
LinearSystem::create_bvec() const
{
  return HVector(_blayout);
}

HVector
LinearSystem::u_to_un(const HVector & u) const
{
  HVector un = create_unvec();
  if (u.zero())
    return un;

  for (std::size_t i = 0; i < _un_to_u.size(); ++i)
  {
    if (_un_to_u[i] == -1)
      continue;
    un[i] = u[_un_to_u[i]];
  }
  return un;
}

HVector
LinearSystem::g_to_gn(const HVector & g) const
{
  HVector gn = create_gnvec();
  if (g.zero())
    return gn;

  for (std::size_t i = 0; i < _gn_to_g.size(); ++i)
  {
    if (_gn_to_g[i] == -1)
      continue;
    gn[i] = g[_gn_to_g[i]];
  }
  return gn;
}

HMatrix
LinearSystem::un_to_u(const HMatrix & A) const
{
  HMatrix Ap(A.block_row_sizes(), shapes_to_shape_refs(_ulayout));
  if (A.zero())
    return Ap;

  for (std::size_t i = 0; i < A.m(); ++i)
  {
    for (std::size_t j = 0; j < _ulayout.size(); ++j)
    {
      auto jp = _u_to_un[j];
      if (jp == -1)
        continue;
      const auto & Aij = A(i, jp);
      if (!Aij.defined())
        continue;
      Ap(i, j) = Aij;
    }
  }
  return Ap;
}

HMatrix
LinearSystem::A()
{
  HMatrix A;
  assemble(&A, nullptr);
  _A_up_to_date = true;
  return A;
}

HVector
LinearSystem::b()
{
  HVector b;
  assemble(nullptr, &b);
  _b_up_to_date = true;
  return b;
}

std::tuple<HMatrix, HVector>
LinearSystem::A_and_b()
{
  HVector b;
  HMatrix A;
  assemble(&A, &b);
  _A_up_to_date = true;
  _b_up_to_date = true;
  return {A, b};
}

} // namespace neml2
