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

#include "neml2/csrc/aoti/nonlinear_system_eager.h"
#include "neml2/csrc/aoti/nonlinear_system.h"

#include <utility>

#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

namespace neml2::aoti
{
namespace py = pybind11;

namespace
{
// NonlinearSystem backed by two Python callables. The givens + linear solver
// are already bound inside the callables (RHS / (Jacobian -> LinearSolve)), so this only ever
// forwards the unknowns. The GIL is held by the caller for the whole solve, so
// the Python calls happen inline with no extra acquire/release.
class EagerNonlinearSystem : public NonlinearSystem
{
public:
  EagerNonlinearSystem(py::object residual_fn,
                       py::object step_fn,
                       std::vector<GroupLayout> unknown_layout,
                       std::vector<GroupLayout> residual_layout)
    : _residual_fn(std::move(residual_fn)),
      _step_fn(std::move(step_fn)),
      _unknown_layout(std::move(unknown_layout)),
      _residual_layout(std::move(residual_layout))
  {
  }

  std::vector<at::Tensor> residual(const std::vector<at::Tensor> & u) const override
  {
    return _residual_fn(u).cast<std::vector<at::Tensor>>();
  }

  std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
  step(const std::vector<at::Tensor> & u) const override
  {
    return _step_fn(u).cast<std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>>();
  }

  const std::vector<GroupLayout> & unknown_layout() const override { return _unknown_layout; }
  const std::vector<GroupLayout> & residual_layout() const override { return _residual_layout; }

private:
  py::object _residual_fn;
  py::object _step_fn;
  std::vector<GroupLayout> _unknown_layout;
  std::vector<GroupLayout> _residual_layout;
};

std::vector<GroupLayout>
to_layouts(const std::vector<std::pair<std::string, std::vector<int64_t>>> & spec)
{
  std::vector<GroupLayout> out;
  out.reserve(spec.size());
  for (const auto & [structure, sub_batch_shape] : spec)
    out.push_back(GroupLayout{structure, sub_batch_shape});
  return out;
}
} // namespace

std::tuple<std::vector<at::Tensor>, bool, std::size_t, std::vector<std::string>>
run_eager_newton(const SolverConfig & cfg,
                 py::object residual_fn,
                 py::object step_fn,
                 std::vector<std::pair<std::string, std::vector<int64_t>>> unknown_layout,
                 std::vector<std::pair<std::string, std::vector<int64_t>>> residual_layout,
                 std::vector<at::Tensor> u0)
{
  EagerNonlinearSystem sys(std::move(residual_fn),
                           std::move(step_fn),
                           to_layouts(unknown_layout),
                           to_layouts(residual_layout));
  NewtonResult result = Newton(cfg).solve(sys, u0);
  return {std::move(result.u), result.converged, result.iterations, std::move(result.log)};
}
} // namespace neml2::aoti
