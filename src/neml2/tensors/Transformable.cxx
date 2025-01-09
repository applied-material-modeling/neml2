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

#include "neml2/tensors/Transformable.h"

#include "neml2/tensors/tensors.h"

namespace neml2
{

R2
transform_from_quaternion(const Quaternion & q)
{
  return q.to_R2();
}

R2
identity_transform(const TensorOptions & options)
{
  return R2::identity(options);
}

R2
proper_rotation_transform(const Rot & rot)
{
  return rot.euler_rodrigues();
}

R2
improper_rotation_transform(const Rot & rot)
{
  Vec v = rot / rot.norm();
  return rot.euler_rodrigues() * (R2::identity(rot.options()) - 2 * v.outer(v));
}

R2
reflection_transform(const Vec & v)
{
  return R2::identity(v.options()) - 2 * v.outer(v);
}

R2
inversion_transform(const TensorOptions & option)
{
  return R2::fill(-1.0, option);
}

} // namespace neml2
