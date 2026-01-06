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

#include "neml2/solvers/NonlinearSystem.h"

namespace neml2
{
OptionSet
NonlinearSystem::expected_options()
{
  OptionSet options;

  return options;
}

NonlinearSystem::NonlinearSystem(const OptionSet & /*options*/) {}

void
NonlinearSystem::set_unknown_ordering(const std::vector<LabeledAxisAccessor> & unknowns)
{
  _unknowns = unknowns;
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::unknown_ordering() const
{
  return _unknowns;
}

void
NonlinearSystem::set_residual_ordering(const std::vector<LabeledAxisAccessor> & residuals)
{
  _residuals = residuals;
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::residual_ordering() const
{
  return _residuals;
}

void
NonlinearSystem::set_prescribed_ordering(const std::vector<LabeledAxisAccessor> & prescribeds)
{
  _prescribeds = prescribeds;
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::prescribed_ordering() const
{
  return _prescribeds;
}

void
NonlinearSystem::set_solution(const es::Vector & x)
{
  set_solution(x);
}

es::Vector
NonlinearSystem::residual()
{
  es::Vector r;
  assemble(&r, nullptr);
  return r;
}

es::Vector
NonlinearSystem::residual(const es::Vector & x)
{
  set_solution(x);
  return residual();
}

es::Matrix
NonlinearSystem::Jacobian()
{
  es::Matrix J;
  assemble(nullptr, &J);
  return J;
}

es::Matrix
NonlinearSystem::Jacobian(const es::Vector & x)
{
  set_solution(x);
  return Jacobian();
}

std::tuple<es::Vector, es::Matrix>
NonlinearSystem::residual_and_Jacobian()
{
  es::Vector r;
  es::Matrix J;
  assemble(&r, &J);
  return {r, J};
}

std::tuple<es::Vector, es::Matrix>
NonlinearSystem::residual_and_Jacobian(const es::Vector & x)
{
  set_solution(x);
  return residual_and_Jacobian();
}
} // namespace neml2
