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

#include "neml2/csrc/aoti/Exception.h"

#include <sstream>
#include <string>
#include <utility>

// Each `recoverable()` is defined out-of-line here so it serves as the class's
// key function: the vtable + typeinfo are emitted in this TU (inside the shared
// library, with AOTI_EXPORT visibility) rather than weakly in every includer.
// That single, exported typeinfo is what lets a downstream module's
// `catch (const neml2::aoti::Exception &)` match an exception thrown across the
// libneml2.so boundary.
namespace neml2::aoti
{
namespace
{
/// Rethrow @p e to inspect it: recoverable iff it is a neml2 Exception that says
/// so. A foreign exception (not ours) is treated as fatal -- we cannot reason
/// about whether retrying it is safe, so we conservatively refuse to recover.
bool
eptr_recoverable(const std::exception_ptr & e)
{
  try
  {
    std::rethrow_exception(e);
  }
  catch (const Exception & ex)
  {
    return ex.recoverable();
  }
  catch (...)
  {
  }
  return false;
}

/// Best-effort message extraction for the aggregate summary.
std::string
eptr_what(const std::exception_ptr & e)
{
  try
  {
    std::rethrow_exception(e);
  }
  catch (const std::exception & ex)
  {
    return ex.what();
  }
  catch (...)
  {
  }
  return "unknown (non-std) exception";
}

std::string
build_aggregate_message(const std::vector<std::exception_ptr> & errors)
{
  std::ostringstream oss;
  oss << errors.size() << " concurrent dispatches failed:";
  for (std::size_t i = 0; i < errors.size(); ++i)
    oss << "\n  [" << i << "] (" << (eptr_recoverable(errors[i]) ? "recoverable" : "fatal") << ") "
        << eptr_what(errors[i]);
  return oss.str();
}
} // namespace

bool
Exception::recoverable() const noexcept
{
  return false;
}

bool
FatalError::recoverable() const noexcept
{
  return false;
}

ConvergenceError::ConvergenceError(const std::string & what_arg,
                                   at::Tensor converged_mask,
                                   std::map<std::string, at::Tensor> unknowns)
  : Exception(what_arg),
    _converged_mask(std::move(converged_mask)),
    _unknowns(std::move(unknowns))
{
}

bool
ConvergenceError::recoverable() const noexcept
{
  return true;
}

AggregateError::AggregateError(std::vector<std::exception_ptr> errors)
  : Exception(build_aggregate_message(errors)),
    _errors(std::move(errors))
{
  // Fatal dominates: recoverable only when every sub-error is recoverable.
  _recoverable = !_errors.empty();
  for (const auto & e : _errors)
    if (!eptr_recoverable(e))
    {
      _recoverable = false;
      break;
    }
}

bool
AggregateError::recoverable() const noexcept
{
  return _recoverable;
}
} // namespace neml2::aoti
