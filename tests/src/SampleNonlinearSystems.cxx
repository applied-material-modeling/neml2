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
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
TestNonlinearSystem::TestNonlinearSystem(std::size_t n)
  : NonlinearSystem(NonlinearSystem::expected_options()),
    _n(n)
{
}

std::vector<LabeledAxisAccessor>
TestNonlinearSystem::setup_umap()
{
  std::vector<LabeledAxisAccessor> umap(_n);
  for (std::size_t i = 0; i < _n; i++)
    umap[i] = LabeledAxisAccessor(STATE, "u" + std::to_string(i));
  return umap;
}

std::vector<TensorShape>
TestNonlinearSystem::setup_intmd_ulayout()
{
  return std::vector<TensorShape>(_n, TensorShape{});
}

std::vector<TensorShape>
TestNonlinearSystem::setup_ulayout()
{
  return std::vector<TensorShape>(_n, TensorShape{});
}

std::vector<LabeledAxisAccessor>
TestNonlinearSystem::setup_bmap()
{
  std::vector<LabeledAxisAccessor> bmap(_n);
  for (std::size_t i = 0; i < _n; i++)
    bmap[i] = LabeledAxisAccessor(RESIDUAL, "r" + std::to_string(i));
  return bmap;
}

std::vector<TensorShape>
TestNonlinearSystem::setup_intmd_blayout()
{
  return std::vector<TensorShape>(_n, TensorShape{});
}

std::vector<TensorShape>
TestNonlinearSystem::setup_blayout()
{
  return std::vector<TensorShape>(_n, TensorShape{});
}

void
PowerTestSystem::assemble(SparseTensorList * A, SparseTensorList * b)
{
  const auto n = _u.size();
  const auto opts = _u.options();

  if (b)
  {
    b->resize(n);
    for (std::size_t i = 0; i < n; i++)
      (*b)[i] = 1.0 - pow(_u[i], Scalar(double(i) + 1, opts));
  }

  if (A)
  {
    A->resize(n * n);
    for (std::size_t i = 0; i < n; i++)
      (*A)[i * n + i] = (i + 1) * pow(_u[i], Scalar(double(i), opts));
  }
}

SparseTensorList
PowerTestSystem::exact_solution(const SparseTensorList & u) const
{
  const auto n = u.size();
  SparseTensorList sol(n);
  for (std::size_t i = 0; i < n; i++)
    sol[i] = Tensor::ones_like(u[i]);
  return sol;
}

void
RosenbrockTestSystem::assemble(SparseTensorList * A, SparseTensorList * b)
{
  const auto n = _u.size();

  if (b)
  {
    b->resize(n);
    for (std::size_t i = 1; i < n - 1; i++)
      (*b)[i] = -200 * (_u[i] - pow(_u[i - 1], 2.0)) + 400 * (_u[i + 1] - pow(_u[i], 2.0)) * _u[i] +
                2 * (1 - _u[i]);
    (*b)[0] = 400 * _u[0] * (_u[1] - pow(_u[0], 2.0)) + 2 * (1 - _u[0]);
    (*b)[n - 1] = -200.0 * (_u[n - 1] - pow(_u[n - 2], 2.0));
  }

  if (A)
  {
    A->resize(n * n);
    for (std::size_t i = 1; i < n - 1; i++)
    {
      (*A)[i * n + i - 1] = -400 * _u[i - 1];
      (*A)[i * n + i] = 202 + 1200 * pow(_u[i], 2.0) - 400 * _u[i + 1];
      (*A)[i * n + i + 1] = -400 * _u[i];
    }
    (*A)[0 * n + 0] = 1200 * pow(_u[0], 2.0) - 400 * _u[1] + 2;
    (*A)[0 * n + 1] = -400 * _u[0];
    (*A)[(n - 1) * n + (n - 2)] = -400 * _u[n - 2];
    (*A)[(n - 1) * n + (n - 1)] = Scalar(200.0, _u.options());
  }
}

SparseTensorList
RosenbrockTestSystem::exact_solution(const SparseTensorList & u) const
{
  const auto n = u.size();
  SparseTensorList sol(n);
  for (std::size_t i = 0; i < n; i++)
    sol[i] = Tensor::ones_like(u[i]);
  return sol;
}
}
