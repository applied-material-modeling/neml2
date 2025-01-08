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

#include <functional>

#include "neml2/dispatcher/WorkGenerator.h"
#include "neml2/dispatcher/WorkScheduler.h"
#include "neml2/misc/error.h"
#include "neml2/misc/types.h"

// Pre-C++20 workaround for std::type_identity
// https://en.cppreference.com/w/cpp/types/type_identity
template <class T>
struct type_identity
{
  using type = T;
};

namespace neml2
{
/**
 * @brief A simple work dispatcher which dispatches work sequentially without asynchronous execution
 *
 * The work dispatcher is responsible for dispatching work generated by a work generator. The
 * dispatcher also takes care of preprocessing, postprocessing, and reducing the work. In general,
 * each work dispatch involves four steps:
 *
 * 1. Work generation: The work generator generates the next \p n batches of work.
 * 2. Preprocessing: The dispatcher preprocesses the work.
 * 3. Do work: The dispatcher dispatches the work to a worker, and the worker complets and returns
 *    the result.
 * 4. Postprocessing: The dispatcher postprocesses the result.
 *
 * Once all the work has been completed and results have been collected, the dispatcher reduces the
 * results to obtain the final result.
 *
 * @tparam I Input type of the preprocessed work (generated by the generator)
 * @tparam O Output type of the result returned by the worker
 * @tparam Of Output type of the final result (after reduction)
 * @tparam Ip Input type of the work before preprocessing
 * @tparam Op Output type of the result after postprocessing
 */
template <typename I,
          typename O,
          typename Of = typename std::vector<O>,
          typename Ip = typename type_identity<I>::type,
          typename Op = typename type_identity<O>::type>
class WorkDispatcher
{
public:
  WorkDispatcher(std::function<O(I &&, torch::Device)> && dispatch)
    : _dispatch(std::move(dispatch))
  {
  }

  WorkDispatcher(std::function<O(I &&, torch::Device)> && dispatch,
                 std::function<O(std::vector<O> &&)> && reduce)
    : _dispatch(std::move(dispatch)),
      _reduce(std::move(reduce))
  {
  }

  WorkDispatcher(std::function<O(I &&, torch::Device)> && dispatch,
                 std::function<Of(std::vector<Op> &&)> && reduce,
                 std::function<I(Ip &&, torch::Device)> && preprocess,
                 std::function<Op(O &&)> && postprocess)
    : _dispatch(std::move(dispatch)),
      _reduce(std::move(reduce)),
      _preprocess(std::move(preprocess)),
      _postprocess(std::move(postprocess))
  {
  }

  Of run(WorkGenerator<Ip> &, WorkScheduler &) const;

protected:
  /// Function to dispatch preprocessed work to the worker and retrieve the result
  std::function<O(Ip &&, torch::Device)> _dispatch;

  /// Function to reduce the results
  std::function<Of(std::vector<Op> &&)> _reduce;

  /// Function to preprocess the work
  std::function<I(Ip &&, torch::Device)> _preprocess;

  /// Function to postprocess the result
  std::function<Op(O &&)> _postprocess;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename I, typename O, typename Of, typename Ip, typename Op>
Of
WorkDispatcher<I, O, Of, Ip, Op>::run(WorkGenerator<Ip> & generator,
                                      WorkScheduler & scheduler) const
{
  neml_assert(bool(_dispatch), "Dispatch function is not set");

  std::vector<Op> results;
  while (generator.has_more())
  {
    // Get the next device and batch size
    auto device = scheduler.next_device();
    auto n = scheduler.next_batch_size();
    neml_assert(n > 0, "Scheduler returned a batch size of ", n);

    // Generate and preprocess work
    // Note that these steps are done _before_ checking if the worker is available. This is
    // because the worker may be available immediately after the work is generated and preprocessed,
    // and we want to dispatch the work as soon as possible.
    auto && [m, work] = generator.next(n);
    if (_preprocess)
      work = _preprocess(std::move(work), device);
    else
      neml_assert(std::is_same_v<I, Ip>, "Preprocess function is not set");

    // Wait until the worker is available to take m batches of work
    while (!scheduler.is_available(device, m))
      // TODO: In theory this could hang if the worker pool is overloaded and no work is completed.
      // We may want to add a timeout here.
      continue;

    // Dispatch
    auto result = _dispatch(std::move(work), device);

    // Tell the scheduler that we have dispatched m batches. Since there is no asynchronous
    // execution, we also immediately tell the scheduler that we have completed m batches
    scheduler.dispatched(device, m);
    scheduler.completed(device, m);

    // Postprocess
    if (_postprocess)
      result = _postprocess(std::move(result));
    else
      neml_assert(std::is_same_v<O, Op>, "Postprocess function is not set");

    results.push_back(result);
  }

  if (_reduce)
    return _reduce(std::move(results));
  else
    neml_assert(std::is_same_v<Of, std::vector<Op>>, "Reduce function is not set");

  if constexpr (std::is_same<Of, std::vector<Op>>::value)
    return results;

  throw NEMLException("Internal error: unreachable code");
}
} // namespace neml2
