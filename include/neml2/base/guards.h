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

#include <string>
#include <map>
#include <chrono>

#include "neml2/misc/types.h"

namespace neml2
{
// Timed section
std::map<std::string, std::map<std::string, unsigned long>> & timed_sections();

struct TimedSection
{
  TimedSection(std::string name, std::string section);

  TimedSection(const TimedSection &) = delete;
  TimedSection(TimedSection &&) = delete;
  TimedSection & operator=(const TimedSection &) = delete;
  TimedSection & operator=(TimedSection &&) = delete;
  ~TimedSection();

private:
  const std::string _name;
  const std::string _section;
  const std::chrono::time_point<std::chrono::high_resolution_clock> _t0;
};

// Set number of interop threads for a local region
struct InterOpThread
{
  InterOpThread(int num);

  InterOpThread(const InterOpThread &) = delete;
  InterOpThread(InterOpThread &&) = delete;
  InterOpThread & operator=(const InterOpThread &) = delete;
  InterOpThread & operator=(InterOpThread &&) = delete;
  ~InterOpThread();

  int prev_num;
};

// Set number of intraop threads for a local region
struct IntraOpThread
{
  IntraOpThread(int num);

  IntraOpThread(const IntraOpThread &) = delete;
  IntraOpThread(IntraOpThread &&) = delete;
  IntraOpThread & operator=(const IntraOpThread &) = delete;
  IntraOpThread & operator=(IntraOpThread &&) = delete;
  ~IntraOpThread();

  int prev_num;
};

}
