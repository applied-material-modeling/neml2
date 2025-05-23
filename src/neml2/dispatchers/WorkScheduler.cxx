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

#include "neml2/dispatchers/WorkScheduler.h"
#include "neml2/base/TracingInterface.h"

namespace neml2
{

OptionSet
WorkScheduler::expected_options()
{
  OptionSet options = NEML2Object::expected_options();
  options += TracingInterface::expected_options();
  options.section() = "Schedulers";

  return options;
}

WorkScheduler::WorkScheduler(const OptionSet & options)
  : NEML2Object(options),
    TracingInterface(options)
{
}

#ifdef NEML2_HAS_JSON
static json
to_json(const Device & device, const std::size_t & batch_size)
{
  json j;
  j["device"] = utils::stringify(device);
  j["batch size"] = batch_size;
  return j;
}
#endif

void
WorkScheduler::schedule_work(Device & device, std::size_t & batch_size)
{
  std::unique_lock<std::mutex> lock(_mutex);

#ifdef NEML2_HAS_JSON
  if (event_tracing_enabled())
    event_trace_writer().trace_duration_begin("schedule work", "WorkScheduler");
#endif

  if (schedule_work_impl(device, batch_size))
  {
#ifdef NEML2_HAS_JSON
    if (event_tracing_enabled())
      event_trace_writer().trace_duration_end(
          "schedule work", "WorkScheduler", to_json(device, batch_size), 0);
#endif
    return;
  }
  _condition.wait(lock,
                  [this, &device, &batch_size] { return schedule_work_impl(device, batch_size); });

#ifdef NEML2_HAS_JSON
  if (event_tracing_enabled())
    event_trace_writer().trace_duration_end(
        "schedule work", "WorkScheduler", to_json(device, batch_size), 0);
#endif
}

void
WorkScheduler::dispatched_work(Device device, std::size_t m)
{
#ifdef NEML2_HAS_JSON
  if (event_tracing_enabled())
    event_trace_writer().trace_duration_begin("dispatch work", "WorkScheduler");
#endif

  std::lock_guard<std::mutex> lock(_mutex);
  dispatched_work_impl(device, m);

#ifdef NEML2_HAS_JSON
  if (event_tracing_enabled())
    event_trace_writer().trace_duration_end("dispatch work", "WorkScheduler", to_json(device, m));
#endif
}

void
WorkScheduler::completed_work(Device device, std::size_t m)
{
#ifdef NEML2_HAS_JSON
  if (event_tracing_enabled())
    event_trace_writer().trace_instant("completed work", "WorkScheduler", to_json(device, m));
#endif
  std::lock_guard<std::mutex> lock(_mutex);
  completed_work_impl(device, m);
  _condition.notify_all();
}

void
WorkScheduler::wait_for_completion()
{
  std::unique_lock<std::mutex> lock(_mutex);
  if (all_work_completed())
  {
#ifdef NEML2_HAS_JSON
    if (event_tracing_enabled())
      event_trace_writer().trace_instant("all work completed", "WorkScheduler");
#endif
    return;
  }
  _condition.wait(lock, [this] { return all_work_completed(); });

#ifdef NEML2_HAS_JSON
  if (event_tracing_enabled())
    event_trace_writer().trace_instant("all work completed", "WorkScheduler");
#endif
}
}
