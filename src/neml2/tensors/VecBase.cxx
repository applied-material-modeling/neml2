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

#include "neml2/tensors/VecBase.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/R3.h"

namespace neml2
{
template <class Derived>
Derived
VecBase<Derived>::rotate(const Rot & r) const
{
  return this->rotate(r.euler_rodrigues());
}

template <class Derived>
Derived
VecBase<Derived>::rotate(const R2 & R) const
{
  return Derived(R * Vec(*this));
}

template <class Derived>
R2
VecBase<Derived>::drotate(const Rot & r) const
{
  return R2::einsum("...ijk,...j", {r.deuler_rodrigues(), *this});
}

template <class Derived>
R3
VecBase<Derived>::drotate(const R2 & R) const
{
  auto I = R2::identity(R.options());
  return R3::einsum("...ij,...k", {I, *this});
}

#define VECBASE_INSTANTIATE(T) template class VecBase<T>
FOR_ALL_VECBASE(VECBASE_INSTANTIATE);
} // namespace neml2
