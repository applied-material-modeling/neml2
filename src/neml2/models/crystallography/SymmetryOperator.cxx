// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/models/crystallography/SymmetryOperator.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/Quaternion.h"

namespace neml2
{
namespace crystallography
{

SymmetryOperator
SymmetryOperator::from_quaternion(const Quaternion & q)
{
  return q.to_R2();
}

SymmetryOperator
SymmetryOperator::Identity(const torch::TensorOptions & options)
{
  return identity(options);
}

SymmetryOperator
SymmetryOperator::ProperRotation(const Rot & rot)
{
  return rot.euler_rodrigues();
}

SymmetryOperator
SymmetryOperator::ImproperRotation(const Rot & rot)
{
  return rot.euler_rodrigues() * (R2::identity(rot.options()) - 2 * rot.outer(rot));
}

SymmetryOperator
SymmetryOperator::Reflection(const Vec & v)
{
  return R2::identity(v.options()) - 2 * v.outer(v);
}

SymmetryOperator
SymmetryOperator::Inversion(const torch::TensorOptions & option)
{
  return R2::fill(-1.0, option);
}

SymmetryOperator
operator*(const SymmetryOperator & A, const SymmetryOperator & B)
{
  return A * B;
}

} // namespace crystallography
} // namespace neml2