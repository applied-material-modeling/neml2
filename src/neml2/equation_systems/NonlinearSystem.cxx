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

#include "neml2/equation_systems/NonlinearSystem.h"
#include "neml2/misc/assertions.h"
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/equation_systems/SparseTensorList.h"

namespace neml2
{
void
NonlinearSystem::setup()
{
  LinearSystem::setup();

  _gmap = setup_gmap();
  _glayout = setup_glayout();
  _intmd_glayout = setup_intmd_glayout();
}

void
NonlinearSystem::u_changed()
{
  _A_up_to_date = false;
  _b_up_to_date = false;
}

void
NonlinearSystem::g_changed()
{
  _A_up_to_date = false;
  _b_up_to_date = false;
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::gmap() const
{
  return _gmap;
}

const std::vector<TensorShape> &
NonlinearSystem::intmd_glayout() const
{
  neml_assert(_intmd_glayout.has_value(),
              "Intermediate shapes for given variables requested but not set up.");
  return _intmd_glayout.value();
}

const std::vector<TensorShape> &
NonlinearSystem::glayout() const
{
  return _glayout;
}

std::tuple<SparseTensorList, SparseTensorList>
NonlinearSystem::A_and_B()
{
  throw NEMLException("A_and_B() not implemented for this NonlinearSystem.");
}

std::tuple<SparseTensorList, SparseTensorList, SparseTensorList>
NonlinearSystem::A_and_B_and_b()
{
  throw NEMLException("A_and_B() not implemented for this NonlinearSystem.");
}

void
NonlinearSystem::post_assemble(bool A, bool b)
{
  LinearSystem::post_assemble(A, b);

  if (!_intmd_glayout.has_value())
    _intmd_glayout = setup_intmd_glayout();
}

std::size_t
NonlinearSystem::p() const
{
  return _gmap.size();
}

} // namespace neml2
