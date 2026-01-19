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
#include "neml2/solvers/HVector.h"
#include "neml2/solvers/HMatrix.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
TestNonlinearSystem::TestNonlinearSystem(const OptionSet & options)
  : NonlinearSystem(options)
{
}

void
TestNonlinearSystem::set_solution(const HVector & x)
{
  _x = x;
}

PowerTestSystem::PowerTestSystem(const OptionSet & options)
  : TestNonlinearSystem(options)
{
}

void
PowerTestSystem::assemble(HVector * residual, HMatrix * Jacobian)
{
  std::vector<TensorShape> s(_x.n(), TensorShape{});

  if (residual)
  {
    *residual = HVector(s);
    for (Size i = 0; i < Size(_x.n()); i++)
      (*residual)[i] = pow(_x[i], Scalar(i + 1, _x.options())) - 1.0;
  }

  if (Jacobian)
  {
    *Jacobian = HMatrix(s, s);
    for (Size i = 0; i < Size(_x.n()); i++)
      (*Jacobian)(i, i) = (i + 1) * pow(_x[i], Scalar(i, _x.options()));
  }
}

HVector
PowerTestSystem::exact_solution(const HVector & x) const
{
  std::vector<TensorShape> s(_x.n(), TensorShape{});
  std::vector<Tensor> sol(x.n());
  for (std::size_t i = 0; i < x.n(); i++)
    sol[i] = Tensor::ones_like(x[i]);
  return HVector(sol, s);
}

RosenbrockTestSystem::RosenbrockTestSystem(const OptionSet & options)
  : TestNonlinearSystem(options)
{
}

void
RosenbrockTestSystem::assemble(HVector * residual, HMatrix * Jacobian)
{
  std::vector<TensorShape> s(_x.n(), TensorShape{});

  if (residual)
  {
    *residual = HVector(s);
    for (Size i = 1; i < Size(_x.n()) - 1; i++)
      (*residual)[i] = 200 * (_x[i] - pow(_x[i - 1], 2.0)) -
                       400 * (_x[i + 1] - pow(_x[i], 2.0)) * _x[i] - 2 * (1 - _x[i]);
    (*residual)[0] = -400 * _x[0] * (_x[1] - pow(_x[0], 2.0)) - 2 * (1 - _x[0]);
    (*residual)[_x.n() - 1] = 200.0 * (_x[_x.n() - 1] - pow(_x[_x.n() - 2], 2.0));
  }

  if (Jacobian)
  {
    *Jacobian = HMatrix(s, s);
    for (Size i = 1; i < Size(_x.n()) - 1; i++)
    {
      (*Jacobian)(i, i - 1) = -400 * _x[i - 1];
      (*Jacobian)(i, i) = 202 + 1200 * pow(_x[i], 2.0) - 400 * _x[i + 1];
      (*Jacobian)(i, i + 1) = -400 * _x[i];
    }
    (*Jacobian)(0, 0) = 1200 * pow(_x[0], 2.0) - 400 * _x[1] + 2;
    (*Jacobian)(0, 1) = -400 * _x[0];
    (*Jacobian)(_x.n() - 1, _x.n() - 2) = -400 * _x[_x.n() - 2];
    (*Jacobian)(_x.n() - 1, _x.n() - 1) = Scalar(200.0, _x.options());
  }
}

HVector
RosenbrockTestSystem::exact_solution(const HVector & x) const
{
  std::vector<TensorShape> s(_x.n(), TensorShape{});
  std::vector<Tensor> sol(x.n());
  for (std::size_t i = 0; i < x.n(); i++)
    sol[i] = Tensor::ones_like(x[i]);
  return HVector(sol, s);
}
}
