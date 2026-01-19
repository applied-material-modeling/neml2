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

#include "neml2/solvers/LinearSystem.h"

namespace neml2
{
LinearSystem::LinearSystem(const std::vector<LabeledAxisAccessor> & umap,
                           const std::vector<TensorShapeRef> & ulayout,
                           const std::vector<LabeledAxisAccessor> & unmap,
                           const std::vector<TensorShapeRef> & unlayout,
                           const std::vector<LabeledAxisAccessor> & gmap,
                           const std::vector<TensorShapeRef> & glayout,
                           const std::vector<LabeledAxisAccessor> & gnmap,
                           const std::vector<TensorShapeRef> & gnlayout,
                           const std::vector<LabeledAxisAccessor> & rmap,
                           const std::vector<TensorShapeRef> & rlayout)
  : _umap(umap),
    _ulayout{ulayout.begin(), ulayout.end()},
    _unmap(unmap),
    _unlayout{unlayout.begin(), unlayout.end()},
    _gmap(gmap),
    _glayout{glayout.begin(), glayout.end()},
    _gnmap(gnmap),
    _gnlayout{gnlayout.begin(), gnlayout.end()},
    _rmap(rmap),
    _rlayout{rlayout.begin(), rlayout.end()}
{
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
LinearSystem::rmap() const
{
  return _rmap;
}

const std::vector<TensorShape> &
LinearSystem::rlayout() const
{
  return _rlayout;
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
LinearSystem::create_rvec() const
{
  return HVector(_rlayout);
}

HMatrix
LinearSystem::A()
{
  HMatrix A;
  assemble(&A, nullptr);
  return A;
}

HVector
LinearSystem::b()
{
  HVector b;
  assemble(nullptr, &b);
  return b;
}

std::tuple<HMatrix, HVector>
LinearSystem::A_and_b()
{
  HVector b;
  HMatrix A;
  assemble(&A, &b);
  return {A, b};
}

} // namespace neml2
