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

#include "SampleNonlinearSystems.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/equation_systems/HVector.h"
#include "neml2/equation_systems/HMatrix.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
void
TestNonlinearSystem::set_u(const HVector & u)
{
  _u = u;
}

void
PowerTestSystem::assemble(HMatrix * A, HVector * b)
{
  std::vector<TensorShape> s(_u.n(), TensorShape{});

  if (b)
  {
    *b = HVector(s);
    for (Size i = 0; i < Size(_u.n()); i++)
      (*b)[i] = 1.0 - pow(_u[i], Scalar(i + 1, _u.options()));
  }

  if (A)
  {
    *A = HMatrix(s, s);
    for (Size i = 0; i < Size(_u.n()); i++)
      (*A)(i, i) = (i + 1) * pow(_u[i], Scalar(i, _u.options()));
  }
}

HVector
PowerTestSystem::exact_solution(const HVector & u) const
{
  std::vector<TensorShape> s(_u.n(), TensorShape{});
  std::vector<Tensor> sol(u.n());
  for (std::size_t i = 0; i < u.n(); i++)
    sol[i] = Tensor::ones_like(u[i]);
  return HVector(sol, s);
}

void
RosenbrockTestSystem::assemble(HMatrix * A, HVector * b)
{
  std::vector<TensorShape> s(_u.n(), TensorShape{});

  if (b)
  {
    *b = HVector(s);
    for (Size i = 1; i < Size(_u.n()) - 1; i++)
      (*b)[i] = -200 * (_u[i] - pow(_u[i - 1], 2.0)) + 400 * (_u[i + 1] - pow(_u[i], 2.0)) * _u[i] +
                2 * (1 - _u[i]);
    (*b)[0] = 400 * _u[0] * (_u[1] - pow(_u[0], 2.0)) + 2 * (1 - _u[0]);
    (*b)[_u.n() - 1] = -200.0 * (_u[_u.n() - 1] - pow(_u[_u.n() - 2], 2.0));
  }

  if (A)
  {
    *A = HMatrix(s, s);
    for (Size i = 1; i < Size(_u.n()) - 1; i++)
    {
      (*A)(i, i - 1) = -400 * _u[i - 1];
      (*A)(i, i) = 202 + 1200 * pow(_u[i], 2.0) - 400 * _u[i + 1];
      (*A)(i, i + 1) = -400 * _u[i];
    }
    (*A)(0, 0) = 1200 * pow(_u[0], 2.0) - 400 * _u[1] + 2;
    (*A)(0, 1) = -400 * _u[0];
    (*A)(_u.n() - 1, _u.n() - 2) = -400 * _u[_u.n() - 2];
    (*A)(_u.n() - 1, _u.n() - 1) = Scalar(200.0, _u.options());
  }
}

HVector
RosenbrockTestSystem::exact_solution(const HVector & u) const
{
  std::vector<TensorShape> s(_u.n(), TensorShape{});
  std::vector<Tensor> sol(u.n());
  for (std::size_t i = 0; i < u.n(); i++)
    sol[i] = Tensor::ones_like(u[i]);
  return HVector(sol, s);
}
}
