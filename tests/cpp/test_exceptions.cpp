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

// Unit tests for the recoverable/fatal exception taxonomy. Pure logic -- no
// compiled artifact. Exercises:
//   - the recoverable() classification of each concrete type;
//   - AggregateError's "fatal dominates" aggregation;
//   - _throw / _assert raising a FatalError;
//   - _guarded normalizing a torch c10::LinAlgError into the recoverable
//     ConvergenceError while carrying the original message through;
//   - the Newton solver throwing a *recoverable* ConvergenceError on both
//     failure modes (divergence and max-iterations), driven by a hand-rolled
//     NonlinearSystem so no AOTI package is needed.

#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/aoti/Model.h" // SolverConfig
#include "neml2/csrc/aoti/assertions.h"
#include "neml2/csrc/aoti/newton.h"
#include "neml2/csrc/aoti/nonlinear_system.h"

#include "test_util.h"

using namespace neml2::aoti;

namespace
{
// Capture `e` as an exception_ptr (preserving its dynamic type) for feeding
// AggregateError.
template <typename E>
std::exception_ptr
as_eptr(E e)
{
  try
  {
    throw e;
  }
  catch (...)
  {
    return std::current_exception();
  }
}

// A trivial single-group DENSE nonlinear system with caller-fixed residual + du,
// enough to drive the Newton solver's failure paths. residual() is constant
// (so it never shrinks) and step() returns a fixed du (zero -> u never moves ->
// max-iterations; the residual itself can be non-finite -> divergence).
class FixedSystem : public NonlinearSystem
{
public:
  FixedSystem(at::Tensor residual_val, at::Tensor du_val)
    : _residual(std::move(residual_val)),
      _du(std::move(du_val)),
      _layout{GroupLayout{"dense", {}}}
  {
  }

  std::vector<at::Tensor> residual(const std::vector<at::Tensor> &) const override
  {
    return {_residual};
  }
  std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
  step(const std::vector<at::Tensor> &) const override
  {
    return {{_du}, {_residual}};
  }
  const std::vector<GroupLayout> & unknown_layout() const override { return _layout; }
  const std::vector<GroupLayout> & residual_layout() const override { return _layout; }

private:
  at::Tensor _residual;
  at::Tensor _du;
  std::vector<GroupLayout> _layout;
};
} // namespace

int
main()
{
  const auto f64 = at::TensorOptions().dtype(at::kDouble);

  // --- recoverable() classification of each concrete type ---------------------
  NEML2_CHECK(ConvergenceError("x").recoverable());
  NEML2_CHECK(!FatalError("x").recoverable());

  // A ConvergenceError is catchable through the base, with the bit intact.
  {
    bool recoverable = false;
    try
    {
      throw ConvergenceError("nope");
    }
    catch (const Exception & e)
    {
      recoverable = e.recoverable();
    }
    NEML2_CHECK(recoverable);
  }

  // --- _throw / _assert raise a (non-recoverable) FatalError ------------------
  {
    bool threw = false, recoverable = true;
    try
    {
      _throw("boom ", 42);
    }
    catch (const FatalError & e)
    {
      threw = true;
      recoverable = e.recoverable();
    }
    NEML2_CHECK(threw);
    NEML2_CHECK(!recoverable);
  }
  NEML2_CHECK_THROWS(_assert(false, "nope"));

  // --- _guarded normalizes c10::LinAlgError -> recoverable ConvergenceError --
  // A torch linalg kernel that hits a singular / non-PD matrix throws
  // c10::LinAlgError; the guard must re-throw it as our recoverable
  // ConvergenceError so a time-stepping consumer cuts the step and retries
  // instead of hard-failing. The original message must survive so the end user
  // can still see which factorization failed.
  {
    bool threw = false, recoverable = false, kept_msg = false, is_fatal = false;
    try
    {
      _guarded([] { TORCH_CHECK_LINALG(false, "singular test matrix"); });
    }
    catch (const FatalError &)
    {
      // Would mean the guard mis-classified this as fatal.
      is_fatal = true;
    }
    catch (const ConvergenceError & e)
    {
      threw = true;
      recoverable = e.recoverable();
      kept_msg = std::strstr(e.what(), "singular test matrix") != nullptr;
    }
    NEML2_CHECK(!is_fatal);
    NEML2_CHECK(threw);
    NEML2_CHECK(recoverable);
    NEML2_CHECK(kept_msg);
  }
  // A LinAlgError caught through the base neml2 Exception surface still reports
  // recoverable() -- the vtable + typeinfo of ConvergenceError is what carries
  // the bit across a downstream `catch (const Exception &)`.
  {
    bool recoverable = false;
    try
    {
      _guarded([] { TORCH_CHECK_LINALG(false, "another singular case"); });
    }
    catch (const Exception & e)
    {
      recoverable = e.recoverable();
    }
    NEML2_CHECK(recoverable);
  }
  // A generic std::runtime_error is still classified fatal (only linalg-shaped
  // torch failures are promoted to recoverable; unrelated foreign errors keep
  // their previous behavior).
  {
    bool fatal = false;
    try
    {
      _guarded([]() -> int { throw std::runtime_error("shape mismatch"); });
    }
    catch (const FatalError & e)
    {
      fatal = !e.recoverable();
    }
    NEML2_CHECK(fatal);
  }

  // --- AggregateError: fatal dominates ---------------------------------------
  { // all recoverable -> aggregate is recoverable
    std::vector<std::exception_ptr> errs{as_eptr(ConvergenceError("a")),
                                         as_eptr(ConvergenceError("b"))};
    AggregateError agg(errs);
    NEML2_CHECK(agg.recoverable());
    NEML2_CHECK(agg.errors().size() == 2);
  }
  { // one fatal among recoverables -> aggregate is fatal
    std::vector<std::exception_ptr> errs{as_eptr(ConvergenceError("a")), as_eptr(FatalError("b"))};
    AggregateError agg(errs);
    NEML2_CHECK(!agg.recoverable());
  }
  { // a foreign (non-neml2) exception is treated as fatal
    std::vector<std::exception_ptr> errs{as_eptr(ConvergenceError("a")),
                                         as_eptr(std::runtime_error("b"))};
    AggregateError agg(errs);
    NEML2_CHECK(!agg.recoverable());
  }
  // An AggregateError is itself a neml2 Exception (single catch surface).
  {
    bool caught = false;
    try
    {
      throw AggregateError({as_eptr(FatalError("x"))});
    }
    catch (const Exception &)
    {
      caught = true;
    }
    NEML2_CHECK(caught);
  }

  // --- Newton solve failures throw a recoverable ConvergenceError -------------
  // Divergence: a non-finite residual must be flagged at the first iteration.
  {
    auto inf = at::full({1, 1}, std::numeric_limits<double>::infinity(), f64);
    auto du = at::zeros({1, 1}, f64);
    FixedSystem sys(inf, du);

    bool threw = false, recoverable = false;
    try
    {
      Newton(SolverConfig{}).solve(sys, {at::zeros({1, 1}, f64)});
    }
    catch (const ConvergenceError & e)
    {
      threw = true;
      recoverable = e.recoverable();
    }
    NEML2_CHECK(threw);
    NEML2_CHECK(recoverable);
  }

  // Max-iterations: a constant nonzero residual with a zero step never converges
  // and never diverges, so the loop exhausts miters and throws.
  {
    SolverConfig cfg;
    cfg.miters = 5;
    auto r = at::full({1, 1}, 1.0, f64);
    auto du = at::zeros({1, 1}, f64);
    FixedSystem sys(r, du);

    bool threw = false, recoverable = false;
    try
    {
      Newton(cfg).solve(sys, {at::zeros({1, 1}, f64)});
    }
    catch (const ConvergenceError & e)
    {
      threw = true;
      recoverable = e.recoverable();
    }
    NEML2_CHECK(threw);
    NEML2_CHECK(recoverable);
  }

  return 0;
}
