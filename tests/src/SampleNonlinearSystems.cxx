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
TestNonlinearSystem::TestNonlinearSystem(TensorShape B,
                                         Size n,
                                         std::vector<Size> residual_group_sizes,
                                         std::vector<Size> unknown_group_sizes)
  : _B(std::move(B)),
    _n(n),
    _residual_group_sizes(residual_group_sizes.empty() ? std::vector<Size>{n}
                                                       : std::move(residual_group_sizes)),
    _unknown_group_sizes(unknown_group_sizes.empty() ? std::vector<Size>{n}
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
    for (Size i = 0; i < _unknown_group_sizes[g]; i++, idx++)
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
    for (Size i = 0; i < _residual_group_sizes[g]; i++, idx++)
      vars[g][i] = LabeledAxisAccessor(RESIDUAL, "r_" + std::to_string(idx));
  }
  std::vector<AxisLayout::IStructure> istr(_residual_group_sizes.size(),
                                           AxisLayout::IStructure::DENSE);
  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes, istr);
}

void
TestNonlinearSystem::assemble(AssembledMatrix * A, AssembledMatrix * /*B*/, AssembledVector * b)
{
  if (b)
  {
    for (_I = 0; _I < blayout().ngroup(); _I++)
    {
      auto group_ndof = (Size)blayout().group(_I).nvar();
      b->tensors[_I] = Tensor::zeros(_B, {}, group_ndof, _u.options());
      for (_i = 0; _i < group_ndof; _i++)
      {
        auto res = residual();
        if (res.defined())
          b->tensors[_I].base_index_put_({_i}, res);
      }
    }
  }

  if (A)
  {
    for (_I = 0; _I < blayout().ngroup(); _I++)
      for (_J = 0; _J < ulayout().ngroup(); _J++)
      {
        auto row_group_ndof = (Size)blayout().group(_I).nvar();
        auto col_group_ndof = (Size)ulayout().group(_J).nvar();
        A->tensors[_I][_J] =
            Tensor::zeros(_B, {}, {Size(row_group_ndof), Size(col_group_ndof)}, _u.options());
        for (_i = 0; _i < row_group_ndof; _i++)
          for (_j = 0; _j < col_group_ndof; _j++)
          {
            auto jac = jacobian();
            if (jac.defined())
              A->tensors[_I][_J].base_index_put_({_i, _j}, jac);
          }
      }
  }
}

AssembledVector
PowerTestSystem::exact_solution(const AssembledVector & u) const
{
  AssembledVector sol(ulayout());
  for (std::size_t i = 0; i < u.layout.ngroup(); i++)
    sol.tensors[i] = Tensor::ones_like(u.tensors[i]);
  return sol;
}

Scalar
PowerTestSystem::residual() const
{
  const auto & u = _u.tensors[_I];
  return 1.0 - pow(u.base_index({_i}), Scalar(double(_i) + 1, _u.options()));
}

Scalar
PowerTestSystem::jacobian() const
{
  if (_I != _J || _i != _j)
    return Scalar();
  const auto & u = _u.tensors[_I];
  return (double(_i) + 1) * pow(u.base_index({_i}), Scalar(double(_i), _u.options()));
}

AssembledVector
RosenbrockTestSystem::exact_solution(const AssembledVector & u) const
{
  AssembledVector sol(ulayout());
  for (std::size_t i = 0; i < u.layout.ngroup(); i++)
    sol.tensors[i] = Tensor::ones_like(u.tensors[i]);
  return sol;
}

Scalar
RosenbrockTestSystem::residual() const
{
  const auto & u = _u.tensors[_I];
  const auto m = (Size)_u.layout.group(_I).nvar();
  if (_i == 0)
    return 400 * u.base_index({_i}) * (u.base_index({_i + 1}) - pow(u.base_index({_i}), 2.0)) +
           2 * (1 - u.base_index({_i}));
  else if (_i == m - 1)
    return -200.0 * (u.base_index({_i}) - pow(u.base_index({_i - 1}), 2.0));
  else
    return -200 * (u.base_index({_i}) - pow(u.base_index({_i - 1}), 2.0)) +
           400 * (u.base_index({_i + 1}) - pow(u.base_index({_i}), 2.0)) * u.base_index({_i}) +
           2 * (1 - u.base_index({_i}));
}

Scalar
RosenbrockTestSystem::jacobian() const
{
  if (_I != _J)
    return Scalar();

  const auto & u = _u.tensors[_I];
  const auto m = (Size)_u.layout.group(_I).nvar();

  if (_i == 0 && _j == 0)
    return 1200 * pow(u.base_index({_i}), 2.0) - 400 * u.base_index({_i + 1}) + 2;
  else if (_i == m - 1 && _j == m - 1)
    return Scalar(200.0, u.base_index({_i}).options());
  else if (_j + 1 == _i)
    return -400 * u.base_index({_j});
  else if (_j == _i)
    return 202 + 1200 * pow(u.base_index({_i}), 2.0) - 400 * u.base_index({_i + 1});
  else if (_i + 1 == _j)
    return -400 * u.base_index({_i});

  return Scalar();
}

}
