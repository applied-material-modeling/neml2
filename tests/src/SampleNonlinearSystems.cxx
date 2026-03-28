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
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
TestNonlinearSystem::TestNonlinearSystem(std::size_t n)
  : _n(n)
{
}

std::shared_ptr<AxisLayout>
TestNonlinearSystem::setup_ulayout()
{
  std::vector<std::vector<LabeledAxisAccessor>> vars(1);
  vars[0].resize(_n);
  for (std::size_t i = 0; i < _n; i++)
    vars[0][i] = LabeledAxisAccessor(STATE, "u_" + std::to_string(i));
  auto intmd_shapes = std::vector<TensorShape>(_n, TensorShape{});
  auto base_shapes = std::vector<TensorShape>(_n, TensorShape{});
  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes);
}

std::shared_ptr<AxisLayout>
TestNonlinearSystem::setup_glayout()
{
  std::vector<std::vector<LabeledAxisAccessor>> vars;
  std::vector<TensorShape> intmd_shapes, base_shapes;
  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes);
}

std::shared_ptr<AxisLayout>
TestNonlinearSystem::setup_blayout()
{
  std::vector<std::vector<LabeledAxisAccessor>> vars(1);
  vars[0].resize(_n);
  for (std::size_t i = 0; i < _n; i++)
    vars[0][i] = LabeledAxisAccessor(STATE, "r_" + std::to_string(i));
  auto intmd_shapes = std::vector<TensorShape>(_n, TensorShape{});
  auto base_shapes = std::vector<TensorShape>(_n, TensorShape{});
  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes);
}

void
PowerTestSystem::assemble(SparseMatrix * A, SparseMatrix * /*B*/, SparseVector * b)
{
  const auto opts = _u.options();

  if (b)
  {
    for (std::size_t i = 0; i < _n; i++)
      b->tensors[i] = 1.0 - pow(_u.tensors[i], Scalar(double(i) + 1, opts));
  }

  if (A)
  {
    for (std::size_t i = 0; i < _n; i++)
      A->tensors[i][i] = (double(i) + 1) * pow(_u.tensors[i], Scalar(double(i), opts));
  }
}

SparseVector
PowerTestSystem::exact_solution(const SparseVector & u) const
{
  SparseVector sol(ulayout()->view());
  for (std::size_t i = 0; i < _n; i++)
    sol.tensors[i] = Tensor::ones_like(u.tensors[i]);
  return sol;
}

void
RosenbrockTestSystem::assemble(SparseMatrix * A, SparseMatrix * /*B*/, SparseVector * b)
{
  if (b)
  {
    for (std::size_t i = 1; i < _n - 1; i++)
      b->tensors[i] = -200 * (_u.tensors[i] - pow(_u.tensors[i - 1], 2.0)) +
                      400 * (_u.tensors[i + 1] - pow(_u.tensors[i], 2.0)) * _u.tensors[i] +
                      2 * (1 - _u.tensors[i]);
    b->tensors[0] =
        400 * _u.tensors[0] * (_u.tensors[1] - pow(_u.tensors[0], 2.0)) + 2 * (1 - _u.tensors[0]);
    b->tensors[_n - 1] = -200.0 * (_u.tensors[_n - 1] - pow(_u.tensors[_n - 2], 2.0));
  }

  if (A)
  {
    for (std::size_t i = 1; i < _n - 1; i++)
    {
      A->tensors[i][i - 1] = -400 * _u.tensors[i - 1];
      A->tensors[i][i] = 202 + 1200 * pow(_u.tensors[i], 2.0) - 400 * _u.tensors[i + 1];
      A->tensors[i][i + 1] = -400 * _u.tensors[i];
    }
    A->tensors[0][0] = 1200 * pow(_u.tensors[0], 2.0) - 400 * _u.tensors[1] + 2;
    A->tensors[0][1] = -400 * _u.tensors[0];
    A->tensors[_n - 1][_n - 2] = -400 * _u.tensors[_n - 2];
    A->tensors[_n - 1][_n - 1] = Scalar(200.0, _u.options());
  }
}

SparseVector
RosenbrockTestSystem::exact_solution(const SparseVector & u) const
{
  SparseVector sol(ulayout()->view());
  for (std::size_t i = 0; i < _n; i++)
    sol.tensors[i] = Tensor::ones_like(u.tensors[i]);
  return sol;
}
}
