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
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/equation_systems/SparseMatrix.h"

namespace neml2
{
class TestNonlinearSystem : public NonlinearSystem
{
public:
  TestNonlinearSystem(std::size_t n);

  void set_u(const SparseVector & u) override { _u = u; }
  void set_g(const SparseVector & /*g*/) override {}

  SparseVector u() const override { return _u; }
  SparseVector g() const override { return {}; }

  virtual SparseVector exact_solution(const SparseVector & u) const = 0;

protected:
  std::shared_ptr<AxisLayout> setup_ulayout() override;
  std::shared_ptr<AxisLayout> setup_glayout() override;
  std::shared_ptr<AxisLayout> setup_blayout() override;

  const std::size_t _n;
  SparseVector _u;
};

class PowerTestSystem : public TestNonlinearSystem
{
public:
  using TestNonlinearSystem::TestNonlinearSystem;
  SparseVector exact_solution(const SparseVector &) const override;

protected:
  void assemble(SparseMatrix *, SparseMatrix *, SparseVector *) override;
};

class RosenbrockTestSystem : public TestNonlinearSystem
{
public:
  using TestNonlinearSystem::TestNonlinearSystem;
  SparseVector exact_solution(const SparseVector &) const override;

protected:
  void assemble(SparseMatrix *, SparseMatrix *, SparseVector *) override;
};
}
