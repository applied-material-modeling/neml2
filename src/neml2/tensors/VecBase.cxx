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
#include "neml2/tensors/tensors.h"
#include "neml2/misc/math.h"

namespace neml2
{
template <class Derived>
Derived
VecBase<Derived>::fill(const Real & v1,
                       const Real & v2,
                       const Real & v3,
                       const torch::TensorOptions & options)
{
  return VecBase<Derived>::fill(Scalar(v1, options), Scalar(v2, options), Scalar(v3, options));
}

template <class Derived>
Derived
VecBase<Derived>::fill(const Scalar & v1, const Scalar & v2, const Scalar & v3)
{
  return Derived(torch::stack({v1, v2, v3}, -1), v1.batch_sizes());
}

template <class Derived>
R2
VecBase<Derived>::identity_map(const torch::TensorOptions & options)
{
  return R2::identity(options);
}

template <class Derived>
Scalar
VecBase<Derived>::operator()(Size i) const
{
  return this->base_index({i});
}

template <class Derived>
Scalar
VecBase<Derived>::norm_sq() const
{
  return dot(*this);
}

template <class Derived>
Scalar
VecBase<Derived>::norm() const
{
  return math::sqrt(dot(*this));
}

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
  return R * Vec(*this);
}

template <class Derived>
R2
VecBase<Derived>::drotate(const Rot & r) const
{
  return torch::einsum("...ijk,...j", {r.deuler_rodrigues(), *this});
}

template <class Derived>
R3
VecBase<Derived>::drotate(const R2 & R) const
{
  auto I = R2::identity(R.options());
  return torch::einsum("...ij,...k", {I, *this});
}

#define VECBASE_INSTANTIATE(T) template class VecBase<T>
FOR_ALL_VECBASE(VECBASE_INSTANTIATE);
} // namespace neml2
