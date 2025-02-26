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
#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "neml2/dispatchers/WorkGenerator.h"
#include "neml2/dispatchers/WorkScheduler.h"
#include "neml2/misc/errors.h"
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
 * @brief The work dispatcher who dispatches work to a worker and reduces the results
 *
 * The work dispatcher coordinates with WorkGenerator and WorkScheduler to dispatch work. The work
 * is generated/loaded by the WorkGenerator; the dispatch is scheduled by a WorkScheduler; the
 * dispatching loop is managed by the WorkDispatcher.
 *
 * The dispatcher also takes care of preprocessing, postprocessing, and reducing the work. In
 * general, each work dispatch involves four steps:
 * 1. Work generation: The work generator generates the next \p n batches of work.
 * 2. Preprocessing: The dispatcher preprocesses the work.
 * 3. Do work: The worker performs the work.
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
  WorkDispatcher(WorkScheduler & scheduler, bool async, std::function<O(I &&, Device)> && do_work)
    : _scheduler(scheduler),
      _devices(scheduler.devices()),
      _async(async),
      _do_work(std::move(do_work))
  {
    init_thread_pool();
  }

  WorkDispatcher(WorkScheduler & scheduler,
                 bool async,
                 std::function<O(I &&, Device)> && do_work,
                 std::function<O(std::vector<O> &&)> && reduce)
    : _scheduler(scheduler),
      _devices(scheduler.devices()),
      _async(async),
      _do_work(std::move(do_work)),
      _reduce(std::move(reduce))
  {
    init_thread_pool();
  }

  WorkDispatcher(WorkScheduler & scheduler,
                 bool async,
                 std::function<O(I &&, Device)> && do_work,
                 std::function<Of(std::vector<Op> &&)> && reduce,
                 std::function<I(Ip &&, Device)> && preprocess,
                 std::function<Op(O &&)> && postprocess)
    : _scheduler(scheduler),
      _devices(scheduler.devices()),
      _async(async),
      _do_work(std::move(do_work)),
      _reduce(std::move(reduce)),
      _preprocess(std::move(preprocess)),
      _postprocess(std::move(postprocess))
  {
    init_thread_pool();
  }

  WorkDispatcher(WorkScheduler & scheduler,
                 bool async,
                 std::function<O(I &&, Device)> && do_work,
                 std::function<Of(std::vector<Op> &&)> && reduce,
                 std::function<I(Ip &&, Device)> && preprocess,
                 std::function<Op(O &&)> && postprocess,
                 std::function<void(std::thread::id, Device)> && thread_init)
    : _scheduler(scheduler),
      _devices(scheduler.devices()),
      _async(async),
      _do_work(std::move(do_work)),
      _reduce(std::move(reduce)),
      _preprocess(std::move(preprocess)),
      _postprocess(std::move(postprocess)),
      _thread_init(std::move(thread_init))
  {
    if (_thread_init)
      neml_assert_dbg(
          _async, "Custom thread initialization functor is only supported in asynchronous mode");

    init_thread_pool();
  }

  ~WorkDispatcher() { stop_thread_pool(); }

  /// Run the dispatching loop (calls run_sync or run_async based on the async flag)
  Of run(WorkGenerator<Ip> &);

protected:
  /// Initialize the thread pool
  void init_thread_pool();

  /// Thread pool main function
  void thread_pool_main(const Device &);

  /// Should unlock thread
  bool should_unlock_thread();

  /// Stop the thread pool
  void stop_thread_pool();

  /// Helper function to validate that the dispatcher is properly configured
  void validate() const;

  /// Run the dispatching loop synchronously
  Of run_sync(WorkGenerator<Ip> &);

  /// Run the dispatching loop asynchronously
  Of run_async(WorkGenerator<Ip> &);

  /// Reference to the work scheduler
  WorkScheduler & _scheduler;

  /// Device pool requested by the scheduler
  const std::vector<Device> _devices;

  /// Flag to enable asynchronous execution
  const bool _async;

  /// Function to perform the work and return the result
  std::function<O(I &&, Device)> _do_work;

  /// Function to reduce the results
  std::function<Of(std::vector<Op> &&)> _reduce;

  /// Function to preprocess the work
  std::function<I(Ip &&, Device)> _preprocess;

  /// Function to postprocess the result
  std::function<Op(O &&)> _postprocess;

  /// Function to initialize the thread
  std::function<void(std::thread::id, Device)> _thread_init;

  /// Results to be reduced
  std::vector<Op> _results;

  ///@{
  /// Mutex for the main thread
  std::mutex _mutex;
  /// Condition variable for waiting all tasks to complete
  std::condition_variable _completion_condition;
  /// Thread pool for asynchronous execution
  /// TODO: We are currently assuming each thread is responsible for one device. This may not be
  /// true/optimal in the future.
  std::vector<std::thread> _thread_pool;
  /// Task queue for the thread pool
  std::unordered_map<Device, std::queue<std::function<void()>>> _tasks;
  /// Mutex for the tasks queue
  std::mutex _qmutex;
  /// Condition variable for the tasks queue
  std::condition_variable _thread_condition;
  /// Flag to stop the thread pool
  bool _stop = false;
  ///@}
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
template <typename I, typename O, typename Of, typename Ip, typename Op>
void
WorkDispatcher<I, O, Of, Ip, Op>::init_thread_pool()
{
  if (!_async)
    return;

  auto nthread = _devices.size();
  _thread_pool.reserve(nthread);
  for (std::size_t i = 0; i < nthread; ++i)
  {
    _thread_pool.emplace_back([this, i] { thread_pool_main(_devices[i]); });
    if (_thread_init)
      _thread_init(_thread_pool.back().get_id(), _devices[i]);
  }
}

template <typename I, typename O, typename Of, typename Ip, typename Op>
void
WorkDispatcher<I, O, Of, Ip, Op>::thread_pool_main(const Device & device)
{
  while (true)
  {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(_qmutex);
      _thread_condition.wait(lock, [this, &device] { return _stop || !_tasks[device].empty(); });
      if (_stop && _tasks[device].empty())
        break;
      task = std::move(_tasks[device].front());
      _tasks[device].pop();
    }
    task();
  }
}

template <typename I, typename O, typename Of, typename Ip, typename Op>
void
WorkDispatcher<I, O, Of, Ip, Op>::stop_thread_pool()
{
  if (!_async)
    return;
  {
    std::unique_lock<std::mutex> lock(_qmutex);
    _stop = true;
  }
  _thread_condition.notify_all();
  for (auto & thread : _thread_pool)
    thread.join();
}

template <typename I, typename O, typename Of, typename Ip, typename Op>
Of
WorkDispatcher<I, O, Of, Ip, Op>::run(WorkGenerator<Ip> & generator)
{
  if (_async)
    return run_async(generator);
  return run_sync(generator);
}

template <typename I, typename O, typename Of, typename Ip, typename Op>
void
WorkDispatcher<I, O, Of, Ip, Op>::validate() const
{
  if (!_do_work)
    throw NEMLException("Do-work function is not set");

  if constexpr (!std::is_same_v<I, Ip>)
    if (!_preprocess)
      throw NEMLException("Preprocess function is not set");

  if constexpr (!std::is_same_v<O, Op>)
    if (!_postprocess)
      throw NEMLException("Postprocess function is not set");

  if constexpr (!std::is_same_v<Of, std::vector<Op>>)
    if (!_reduce)
      throw NEMLException("Reduce function is not set");
}

template <typename I, typename O, typename Of, typename Ip, typename Op>
Of
WorkDispatcher<I, O, Of, Ip, Op>::run_sync(WorkGenerator<Ip> & generator)
{
  validate();

  Device device = kCPU;
  std::size_t n = 0;
  _results.clear();
  while (generator.has_more())
  {
    _scheduler.schedule_work(device, n);
    if (n <= 0)
      throw NEMLException("Scheduler returned a batch size of " + std::to_string(n));
    // Generate work
    auto && [m, work] = generator.next(n);
    // Preprocess
    if (_preprocess)
      work = _preprocess(std::move(work), device);
    // Do work. Since there is no asynchronous execution, we do not notify the scheduler (this also
    // avoids potential parallel communication inccured by the scheduler)
    auto result = _do_work(std::move(work), device);
    // Postprocess
    if (_postprocess)
      result = _postprocess(std::move(result));
    _results.push_back(result);
  }

  if (_reduce)
    return _reduce(std::move(_results));

  if constexpr (std::is_same<Of, std::vector<Op>>::value)
    return _results;

  throw NEMLException("Internal error: unreachable code");
}

template <typename I, typename O, typename Of, typename Ip, typename Op>
Of
WorkDispatcher<I, O, Of, Ip, Op>::run_async(WorkGenerator<Ip> & generator)
{
  validate();

  Device device = kCPU;
  std::size_t n = 0;
  _results.clear();

  // Keep asking the scheduler for an available device
  // - If the generator has no more work, we break out of the loop
  // - If the scheduler schedules work, we dispatch the work and continue with the dispatching loop
  while (generator.has_more())
  {
    _scheduler.schedule_work(device, n);
    if (n <= 0)
      throw NEMLException("Scheduler returned a batch size of " + std::to_string(n));
    // Generate work
    auto && [m, work] = generator.next(n);
    // Reserve space for the result
    _results.resize(_results.size() + 1);
    auto i = _results.size() - 1;
    // Create the task
    auto task = [this, work = std::move(work), device = device, m = m, i = i]() mutable
    {
      // Preprocess
      if (_preprocess)
        work = _preprocess(std::move(work), device);
      // Do work
      auto result = _do_work(std::move(work), device);
      // Postprocess
      if (_postprocess)
        result = _postprocess(std::move(result));
      // Collect result
      _results[i] = std::move(result);
      // Tell the scheduler that we have completed m batches
      _scheduler.completed_work(device, m);
    };
    // Tell the scheduler that we have dispatched m batches
    _scheduler.dispatched_work(device, m);
    // Enqueue the task
    {
      std::unique_lock<std::mutex> lock(_qmutex);
      _tasks[device].push(task);
    }
    // Notify the thread pool
    // Note: We notify_all instead of notify_one because we want the thread that's bind to the
    // target device to pick up the task
    _thread_condition.notify_all();
  }

  // Wait for all tasks to complete
  _scheduler.wait_for_completion();

  if (_reduce)
    return _reduce(std::move(_results));

  if constexpr (std::is_same<Of, std::vector<Op>>::value)
    return _results;

  throw NEMLException("Internal error: unreachable code");
}
} // namespace neml2
