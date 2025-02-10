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

#include "neml2/solvers/TrustRegionSubProblem.h"
#include "neml2/tensors/functions/bmm.h"
#include "neml2/tensors/functions/bmv.h"
#include "neml2/tensors/functions/bvv.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/linalg/solve.h"

namespace neml2
{
TrustRegionSubProblem::TrustRegionSubProblem(const OptionSet & options)
  : NonlinearSystem(options)
{
}

void
TrustRegionSubProblem::reinit(const NonlinearSystem::Res<true> & r,
                              const NonlinearSystem::Jac<true> & J,
                              const Scalar & delta)
{
  _delta = delta;
  _JJ = bmm(J.base_transpose(0, 1), J);
  _Jr = bmv(J.base_transpose(0, 1), r);
}

void
TrustRegionSubProblem::set_guess(const NonlinearSystem::Sol<false> & x)
{
  _s = Scalar(x);
}

void
TrustRegionSubProblem::assemble(NonlinearSystem::Res<false> * residual,
                                NonlinearSystem::Jac<false> * Jacobian)
{
  auto p = -preconditioned_direction(_s);
  auto np = sqrt(bvv(p, p));

  if (residual)
    *residual = NonlinearSystem::Res<false>(1.0 / np - 1.0 / sqrt(2.0 * _delta));

  if (Jacobian)
    *Jacobian =
        NonlinearSystem::Jac<false>(1.0 / pow(np, 3.0) * bvv(p, preconditioned_solve(_s, p)));
}

Tensor
TrustRegionSubProblem::preconditioned_solve(const Scalar & s, const Tensor & v) const
{
  return linalg::solve(_JJ + s * Tensor::identity(v.base_sizes()[0], v.options()), v);
}

Tensor
TrustRegionSubProblem::preconditioned_direction(const Scalar & s) const
{
  return preconditioned_solve(s, _Jr);
}
} // namespace neml2
