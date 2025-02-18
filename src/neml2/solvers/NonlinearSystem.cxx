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

#include "neml2/solvers/NonlinearSystem.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/bmm.h"
#include "neml2/tensors/functions/diag_embed.h"

namespace neml2
{
bool &
currently_solving_nonlinear_system()
{
  static bool _solving_nl_sys = false;
  return _solving_nl_sys;
}

SolvingNonlinearSystem::SolvingNonlinearSystem(bool solving)
  : prev_bool(currently_solving_nonlinear_system())
{
  currently_solving_nonlinear_system() = solving;
}

SolvingNonlinearSystem::~SolvingNonlinearSystem()
{
  currently_solving_nonlinear_system() = prev_bool;
}

OptionSet
NonlinearSystem::expected_options()
{
  OptionSet options;

  options.set<bool>("automatic_scaling") = false;
  options.set("automatic_scaling").doc() =
      "Whether to perform automatic scaling. See neml2::NonlinearSystem::init_scaling for "
      "implementation details.";

  options.set<Real>("automatic_scaling_tol") = 0.01;
  options.set("automatic_scaling_tol").doc() =
      "Tolerance used in iteratively updating the scaling matrices.";

  options.set<unsigned int>("automatic_scaling_miter") = 20;
  options.set("automatic_scaling_miter").doc() =
      "Maximum number of automatic scaling iterations. No error is produced upon reaching the "
      "maximum number of scaling iterations, and the scaling matrices obtained at the last "
      "iteration are used to scale the nonlinear system.";

  return options;
}

void
NonlinearSystem::disable_automatic_scaling(OptionSet & options)
{
  options.set("automatic_scaling").suppressed() = true;
  options.set("automatic_scaling_tol").suppressed() = true;
  options.set("automatic_scaling_miter").suppressed() = true;
}

void
NonlinearSystem::enable_automatic_scaling(OptionSet & options)
{
  options.set("automatic_scaling").suppressed() = false;
  options.set("automatic_scaling_tol").suppressed() = false;
  options.set("automatic_scaling_miter").suppressed() = false;
}

NonlinearSystem::NonlinearSystem(const OptionSet & options)
  : _autoscale(options.get<bool>("automatic_scaling")),
    _autoscale_tol(options.get<Real>("automatic_scaling_tol")),
    _autoscale_miter(options.get<unsigned int>("automatic_scaling_miter")),
    _scaling_matrices_initialized(false)
{
}

void
NonlinearSystem::init_scaling(const NonlinearSystem::Sol<false> & x, const bool verbose)
{
  if (!_autoscale)
    return;

  if (_scaling_matrices_initialized)
    return;

  // First compute the unscaled Jacobian
  auto J = Jacobian(x);

  // Initialize the scaling matrices
  auto Jp = J.clone();
  _row_scaling = Tensor::ones_like(x);
  _col_scaling = Tensor::ones_like(x);

  if (verbose)
    std::cout << "Before automatic scaling cond(J) = " << std::scientific
              << at::max(at::linalg_cond(Jp)).item<Real>() << std::endl;

  for (unsigned int itr = 0; itr < _autoscale_miter; itr++)
  {
    // check for convergence
    auto rR = at::max(at::abs(1.0 - 1.0 / at::sqrt(std::get<0>(Jp.max(-1))))).item<Real>();
    auto rC = at::max(at::abs(1.0 - 1.0 / at::sqrt(std::get<0>(Jp.max(-2))))).item<Real>();
    if (verbose)
      std::cout << "ITERATION " << itr << ", ROW ILLNESS = " << std::scientific << rR
                << ", COL ILLNESS = " << std::scientific << rC << std::endl;
    if (rR < _autoscale_tol && rC < _autoscale_tol)
      break;

    // scale rows and columns
    for (Size i = 0; i < x.base_size(-1); i++)
    {
      auto ar = 1.0 / at::sqrt(at::max(Jp.base_index({i})));
      auto ac = 1.0 / at::sqrt(at::max(Jp.base_index({indexing::Slice(), i})));
      _row_scaling.base_index({i}) *= ar;
      _col_scaling.base_index({i}) *= ac;
      Jp.base_index({i}) *= ar;
      Jp.base_index({indexing::Slice(), i}) *= ac;
    }
  }

  _scaling_matrices_initialized = true;

  if (verbose)
    std::cout << " After automatic scaling cond(J) = " << std::scientific
              << at::max(at::linalg_cond(Jp)).item<Real>() << std::endl;
}

void
NonlinearSystem::ensure_scaling_matrices_initialized_dbg() const
{
  neml_assert_dbg(
      _autoscale == _scaling_matrices_initialized,
      _autoscale ? "Automatic scaling is requested but scaling matrices have not been initialized."
                 : "Automatic scaling is not requested but scaling matrices were initialized.");
}

NonlinearSystem::Res<true>
NonlinearSystem::scale(const NonlinearSystem::Res<false> & r) const
{
  if (!_autoscale)
    return Res<true>(r);

  ensure_scaling_matrices_initialized_dbg();
  return Res<true>(_row_scaling * r);
}

NonlinearSystem::Res<false>
NonlinearSystem::unscale(const NonlinearSystem::Res<true> & r) const
{
  if (!_autoscale)
    return Res<false>(r);

  ensure_scaling_matrices_initialized_dbg();
  return Res<false>(1. / _row_scaling * r);
}

NonlinearSystem::Jac<true>
NonlinearSystem::scale(const NonlinearSystem::Jac<false> & J) const
{
  if (!_autoscale)
    return Jac<true>(J);

  ensure_scaling_matrices_initialized_dbg();
  return Jac<true>(bmm(bmm(base_diag_embed(_row_scaling), J), base_diag_embed(_col_scaling)));
}

NonlinearSystem::Jac<false>
NonlinearSystem::unscale(const NonlinearSystem::Jac<true> & J) const
{
  if (!_autoscale)
    return Jac<false>(J);

  ensure_scaling_matrices_initialized_dbg();
  return Jac<false>(
      bmm(bmm(base_diag_embed(1.0 / _row_scaling), J), base_diag_embed(1.0 / _col_scaling)));
}

NonlinearSystem::Sol<true>
NonlinearSystem::scale(const NonlinearSystem::Sol<false> & u) const
{
  if (!_autoscale)
    return Sol<true>(u);

  ensure_scaling_matrices_initialized_dbg();
  return Sol<true>(1. / _col_scaling * u);
}

NonlinearSystem::Sol<false>
NonlinearSystem::unscale(const NonlinearSystem::Sol<true> & u) const
{
  if (!_autoscale)
    return Sol<false>(u);

  ensure_scaling_matrices_initialized_dbg();
  return Sol<false>(_col_scaling * u);
}

void
NonlinearSystem::set_guess(const NonlinearSystem::Sol<true> & x)
{
  set_guess(unscale(x));
}
} // namespace neml2
