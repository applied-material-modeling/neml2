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

#include <iostream>
#include <iomanip>

#include "neml2/solvers/NewtonWithLineSearch.h"
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{
register_NEML2_object(NewtonWithLineSearch);

OptionSet
NewtonWithLineSearch::expected_options()
{
  OptionSet options = Newton::expected_options();
  options.doc() = "The Newton-Raphson solver with line search.";

  EnumSelection linesearch_type({"BACKTRACKING", "STRONG_WOLFE"}, "BACKTRACKING");
  options.set<EnumSelection>("linesearch_type") = linesearch_type;
  options.set("linesearch_type").doc() = "The type of linesearch used."
                                         "Default: BACKTRACKING. Options are " +
                                         linesearch_type.join();

  options.set<unsigned int>("max_linesearch_iterations") = 10;
  options.set("max_linesearch_iterations").doc() =
      "Maximum allowable linesearch iterations. No error is produced upon reaching the maximum "
      "number of iterations, and the scale factor in the last iteration is used to scale the step.";

  options.set<double>("linesearch_cutback") = 2.0;
  options.set("linesearch_cutback").doc() = "Linesearch cut-back factor when the current scale "
                                            "factor cannot sufficiently reduce the residual.";

  options.set<double>("linesearch_stopping_criteria") = 1.0e-3;
  options.set("linesearch_stopping_criteria").doc() =
      "The lineseach tolerance slightly relaxing the definition of residual decrease";

  options.set<bool>("check_negative_critertion") = false;
  options.set("check_negative_critertion").doc() =
      "Whether to check if the threshold used in the convergence criterion for line search becomes "
      "negative. If true, and a negative value is detected, a warning message is printed to cerr.";

  return options;
}

NewtonWithLineSearch::NewtonWithLineSearch(const OptionSet & options)
  : Newton(options),
    _linesearch_miter(options.get<unsigned int>("max_linesearch_iterations")),
    _linesearch_sigma(options.get<double>("linesearch_cutback")),
    _linesearch_c(options.get<double>("linesearch_stopping_criteria")),
    _type(options.get<EnumSelection>("linesearch_type")),
    _check_crit(options.get<bool>("check_negative_critertion"))
{
}

void
NewtonWithLineSearch::update(NonlinearSystem & sys)
{
  auto du = linear_solver->solve(sys);
  auto u = sys.u();
  auto alpha = Scalar::ones(u.options());
  const auto b0 = sys.b();
  const auto nb0 = neml2::norm_sq(b0);
  auto crit = nb0;

  for (std::size_t i = 1; i < _linesearch_miter; i++)
  {
    auto up = u + alpha * du;
    sys.set_u(up);
    auto b = sys.b();
    auto nb = norm_sq(b);

    if (_type == "BACKTRACKING")
      crit = nb0 - 2.0 * _linesearch_c * alpha * neml2::inner(b0, du);
    else if (_type == "STRONG_WOLFE")
      crit = (1.0 - _linesearch_c * alpha) * nb0;

    if (verbose)
      std::cout << "     LS ITERATION " << std::setw(3) << i << ", min(alpha) = " << std::scientific
                << at::min(alpha).item<double>() << ", max(||R||) = " << std::scientific
                << at::max(sqrt(nb)).item<double>() << ", min(||Rc||) = " << std::scientific
                << at::min(sqrt(crit)).item<double>() << std::endl;

    auto stop = nb <= crit || nb <= std::pow(atol, 2);

    if (at::all(stop).item<bool>())
      break;

    alpha = alpha.dynamic_expand_as(stop).contiguous();
    alpha.dynamic_index_put_({!stop}, alpha.dynamic_index({!stop}) / _linesearch_sigma);
  }

  if (_check_crit)
    if (at::max(crit).item<double>() < 0)
      std::cerr << "WARNING: Line Search produces negative stopping criteria, this could lead to "
                   "convergence issue. Try with other linesearch_type, increase linesearch_cutback "
                   "or reduce linesearch_stopping_criteria"
                << std::endl;
}

} // namespace neml2
