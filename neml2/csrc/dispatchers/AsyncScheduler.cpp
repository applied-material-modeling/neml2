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

#include "neml2/csrc/dispatchers/AsyncScheduler.h"

namespace neml2::aoti
{
void
AsyncScheduler::schedule_work(at::Device & dev, std::size_t & n)
{
  std::unique_lock<std::mutex> lock(_mutex);
  _cv.wait(lock, [&] { return schedule_work_impl(dev, n); });
}

void
AsyncScheduler::dispatched_work(at::Device dev, std::size_t n)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    dispatched_work_impl(dev, n);
  }
  _cv.notify_all();
}

void
AsyncScheduler::completed_work(at::Device dev, std::size_t n)
{
  {
    std::lock_guard<std::mutex> lock(_mutex);
    completed_work_impl(dev, n);
  }
  // Frees capacity (wakes schedule_work) and may drain the last load (wakes
  // wait_for_completion).
  _cv.notify_all();
}

void
AsyncScheduler::wait_for_completion()
{
  std::unique_lock<std::mutex> lock(_mutex);
  _cv.wait(lock, [&] { return all_work_completed(); });
}
} // namespace neml2::aoti
