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

#pragma once

#include "neml2/equation_systems/NonlinearSystem.h"
#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/equation_systems/AssembledMatrix.h"

namespace neml2
{
class TestNonlinearSystem : public NonlinearSystem
{
public:
  /// @param n Total number of DOFs.
  /// @param residual_group_sizes Number of DOFs in each residual group.  When empty
  ///        (the default) all DOFs are placed in a single group.
  /// @param unknown_group_sizes Number of DOFs in each unknown group.  When empty
  ///        (the default) all DOFs are placed in a single group.
  TestNonlinearSystem(std::size_t n,
                      std::vector<std::size_t> residual_group_sizes = {},
                      std::vector<std::size_t> unknown_group_sizes = {});

  void set_u(const AssembledVector & u) override { _u = u; }
  void set_g(const AssembledVector & /*g*/) override {}

  AssembledVector u() const override { return _u; }
  AssembledVector g() const override { return {}; }

  virtual AssembledVector exact_solution(const AssembledVector & u) const = 0;

protected:
  void assemble(AssembledMatrix *, AssembledMatrix *, AssembledVector *) override;

  std::shared_ptr<AxisLayout> setup_ulayout() override;
  std::shared_ptr<AxisLayout> setup_glayout() override;
  std::shared_ptr<AxisLayout> setup_blayout() override;

  /// residual for DOF i
  virtual Scalar residual() const = 0;
  /// Jacobian for DOF i w.r.t. DOF j
  virtual Scalar jacobian() const = 0;

  /// DOF index
  Size _i = 0, _j = 0;
  /// group index
  std::size_t _I = 0, _J = 0;

  /// Number of DOFs in the system.
  const Size _n;
  /// DOFs per residual group (always at least one entry)
  const std::vector<std::size_t> _residual_group_sizes;
  /// DOFs per unknown group (always at least one entry)
  const std::vector<std::size_t> _unknown_group_sizes;
  /// Current solution vector.
  AssembledVector _u;
};

class PowerTestSystem : public TestNonlinearSystem
{
public:
  using TestNonlinearSystem::TestNonlinearSystem;
  AssembledVector exact_solution(const AssembledVector &) const override;

protected:
  /// residual for DOF i
  Scalar residual() const override;
  /// Jacobian for DOF i w.r.t. DOF j
  Scalar jacobian() const override;
};

class RosenbrockTestSystem : public TestNonlinearSystem
{
public:
  using TestNonlinearSystem::TestNonlinearSystem;
  AssembledVector exact_solution(const AssembledVector &) const override;

protected:
  /// residual for DOF i
  Scalar residual() const override;
  /// Jacobian for DOF i w.r.t. DOF j
  Scalar jacobian() const override;
};
}
