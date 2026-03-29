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
#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
TestNonlinearSystem::TestNonlinearSystem(std::size_t n,
                                         std::vector<std::size_t> residual_group_sizes,
                                         std::vector<std::size_t> unknown_group_sizes)
  : _n((Size)n),
    _residual_group_sizes(residual_group_sizes.empty() ? std::vector<std::size_t>{n}
                                                       : std::move(residual_group_sizes)),
    _unknown_group_sizes(unknown_group_sizes.empty() ? std::vector<std::size_t>{n}
                                                     : std::move(unknown_group_sizes))
{
}

std::shared_ptr<AxisLayout>
TestNonlinearSystem::setup_ulayout()
{
  std::vector<std::vector<LabeledAxisAccessor>> vars(_unknown_group_sizes.size());
  std::vector<TensorShape> intmd_shapes(_n, TensorShape{});
  std::vector<TensorShape> base_shapes(_n, TensorShape{});
  std::size_t idx = 0;
  for (std::size_t g = 0; g < _unknown_group_sizes.size(); g++)
  {
    vars[g].resize(_unknown_group_sizes[g]);
    for (std::size_t i = 0; i < _unknown_group_sizes[g]; i++, idx++)
      vars[g][i] = LabeledAxisAccessor(STATE, "u_" + std::to_string(idx));
  }
  std::vector<AxisLayout::IStructure> istr(_unknown_group_sizes.size(),
                                           AxisLayout::IStructure::DENSE);
  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes, istr);
}

std::shared_ptr<AxisLayout>
TestNonlinearSystem::setup_glayout()
{
  std::vector<std::vector<LabeledAxisAccessor>> vars;
  std::vector<TensorShape> intmd_shapes, base_shapes;
  std::vector<AxisLayout::IStructure> istr;
  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes, istr);
}

std::shared_ptr<AxisLayout>
TestNonlinearSystem::setup_blayout()
{
  std::vector<std::vector<LabeledAxisAccessor>> vars(_residual_group_sizes.size());
  std::vector<TensorShape> intmd_shapes(_n, TensorShape{});
  std::vector<TensorShape> base_shapes(_n, TensorShape{});
  std::size_t idx = 0;
  for (std::size_t g = 0; g < _residual_group_sizes.size(); g++)
  {
    vars[g].resize(_residual_group_sizes[g]);
    for (std::size_t i = 0; i < _residual_group_sizes[g]; i++, idx++)
      vars[g][i] = LabeledAxisAccessor(STATE, "r_" + std::to_string(idx));
  }
  std::vector<AxisLayout::IStructure> istr(_residual_group_sizes.size(),
                                           AxisLayout::IStructure::DENSE);
  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes, istr);
}

void
TestNonlinearSystem::assemble(AssembledMatrix * A, AssembledMatrix * /*B*/, AssembledVector * b)
{
  const auto opts = _u.options();

  if (b)
  {
    _i = 0;
    for (_I = 0; _I < blayout()->ngroup(); _I++)
    {
      auto group_ndof = (Size)blayout()->group(_I).nvar();
      b->tensors[_I] = Tensor::empty({Size(group_ndof)}, opts);
      for (Size i = 0; i < group_ndof; i++, _i++)
      {
        auto res = residual();
        if (res.defined())
          b->tensors[_I].base_index_put_({i}, res);
      }
    }
  }

  if (A)
  {
    _i = 0;
    _j = 0;
    for (_I = 0; _I < blayout()->ngroup(); _I++)
      for (_J = 0; _J < ulayout()->ngroup(); _J++)
      {
        auto row_group_ndof = (Size)blayout()->group(_I).nvar();
        auto col_group_ndof = (Size)ulayout()->group(_J).nvar();
        A->tensors[_I][_J] = Tensor::empty({Size(row_group_ndof), Size(col_group_ndof)}, opts);
        for (Size i = 0; i < row_group_ndof; i++, _i++)
          for (Size j = 0; j < col_group_ndof; j++, _j++)
          {
            auto jac = jacobian();
            if (jac.defined())
              A->tensors[_I][_J].base_index_put_({i, j}, jac);
          }
      }
  }
}

AssembledVector
PowerTestSystem::exact_solution(const AssembledVector & u) const
{
  AssembledVector sol(ulayout()->view());
  for (Size i = 0; i < _n; i++)
    sol.tensors[i] = Tensor::ones_like(u.tensors[i]);
  return sol;
}

Scalar
PowerTestSystem::residual() const
{
  return 1.0 - pow(_u.tensors[_i], Scalar(double(_i) + 1, _u.options()));
}

Scalar
PowerTestSystem::jacobian() const
{
  if (_i != _j)
    return Scalar();
  return (double(_i) + 1) * pow(_u.tensors[_i], Scalar(double(_i), _u.options()));
}

AssembledVector
RosenbrockTestSystem::exact_solution(const AssembledVector & u) const
{
  AssembledVector sol(ulayout()->view());
  for (Size i = 0; i < _n; i++)
    sol.tensors[i] = Tensor::ones_like(u.tensors[i]);
  return sol;
}

Scalar
RosenbrockTestSystem::residual() const
{
  const auto & u = _u.tensors;
  if (_i == 0)
    return 400 * u[0] * (u[1] - pow(u[0], 2.0)) + 2 * (1 - u[0]);
  else if (_i == _n - 1)
    return -200.0 * (u[_n - 1] - pow(u[_n - 2], 2.0));
  else
    return -200 * (u[_i] - pow(u[_i - 1], 2.0)) + 400 * (u[_i + 1] - pow(u[_i], 2.0)) * u[_i] +
           2 * (1 - u[_i]);
}

Scalar
RosenbrockTestSystem::jacobian() const
{
  const auto & u = _u.tensors;

  if (_i == 0 && _j == 0)
    return 1200 * pow(u[0], 2.0) - 400 * u[1] + 2;
  else if (_i == 0 && _j == 1)
    return -400 * u[0];
  else if (_i == _n - 1 && _j == _n - 2)
    return -400 * u[_n - 2];
  else if (_i == _n - 1 && _j == _n - 1)
    return Scalar(200.0, u[0].options());
  else if (_j == _i - 1)
    return -400 * u[_j];
  else if (_j == _i)
    return 202 + 1200 * pow(u[_i], 2.0) - 400 * u[_i + 1];
  else if (_j == _i + 1)
    return -400 * u[_i];

  return Scalar();
}

}
