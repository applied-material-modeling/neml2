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

#include <mutex>
#include <iostream>
#include <ATen/Parallel.h>

#include "neml2/base/guards.h"

namespace neml2
{
// LCOV_EXCL_START
std::map<std::string, std::map<std::string, unsigned long>> &
timed_sections()
{
  static std::map<std::string, std::map<std::string, unsigned long>> _timed_sections_singleton;
  return _timed_sections_singleton;
}

TimedSection::TimedSection(std::string name, std::string section)
  : _name(std::move(name)),
    _section(std::move(section)),
    _t0(std::chrono::high_resolution_clock::now())
{
}

TimedSection::~TimedSection()
{
  auto t1 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - _t0).count();

  static std::mutex _timed_sections_write_mutex;
  {
    std::scoped_lock lock(_timed_sections_write_mutex);
    timed_sections()[_section][_name] += dt;
  }
}

InterOpThread::InterOpThread(int num)
  : prev_num(at::get_num_interop_threads())
{
  if (num > 0)
    at::set_num_interop_threads(num);
}

InterOpThread::~InterOpThread() { at::set_num_interop_threads(prev_num); }

IntraOpThread::IntraOpThread(int num)
  : prev_num(at::get_num_threads())
{
  if (num > 0)
    at::set_num_threads(num);
}

IntraOpThread::~IntraOpThread() { at::set_num_threads(prev_num); }
// LCOV_EXCL_STOP
}
