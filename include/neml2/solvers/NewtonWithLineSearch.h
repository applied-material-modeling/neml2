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

#include "neml2/solvers/Newton.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
/**
 * @copydoc neml2::Newton
 *
 * Armijo line search strategy is used to search along the direction of the full Newton step for a
 * decreasing residual norm.
 */
class NewtonWithLineSearch : public Newton
{
public:
  static OptionSet expected_options();

  NewtonWithLineSearch(const OptionSet & options);

protected:
  /// Update trial solution
  void update(NonlinearSystem & system,
              NonlinearSystem::Sol<true> & x,
              const NonlinearSystem::Res<true> & r,
              const NonlinearSystem::Jac<true> & J) override;

  /// Perform Armijo linesearch
  virtual Scalar linesearch(NonlinearSystem & system,
                            const NonlinearSystem::Sol<true> & x,
                            const NonlinearSystem::Sol<true> & dx,
                            const NonlinearSystem::Res<true> & R0) const;

  /// Linesearch maximum iterations
  unsigned int _linesearch_miter;

  /// Decrease factor for linesearch
  double _linesearch_sigma;

  /// Stopping criteria for linesearch
  double _linesearch_c;

  /// Seclect the type of line search
  EnumSelection _type;

  bool _check_crit;
};
} // namespace neml2
