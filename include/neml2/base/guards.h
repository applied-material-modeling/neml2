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

#include <ATen/Parallel.h>

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
  TimedSection(const std::string & name, const std::string & section);

  ~TimedSection();

private:
  const std::string _name;
  const std::string _section;
  const std::chrono::time_point<std::chrono::high_resolution_clock> _t0;
};

// Guard a region where implicit solve is being performed
struct SolvingNonlinearSystem
{
  SolvingNonlinearSystem(bool solving = true)
    : prev_bool(currently_solving_nonlinear_system())
  {
    currently_solving_nonlinear_system() = solving;
  }

  ~SolvingNonlinearSystem() { currently_solving_nonlinear_system() = prev_bool; }

  const bool prev_bool;
};

// Set number of interop threads for a local region
struct InterOpThread
{
  InterOpThread(unsigned int num)
    : prev_num(at::get_num_interop_threads())
  {
    if (num > 0)
      at::set_num_interop_threads(num);
  }

  ~InterOpThread() { at::set_num_interop_threads(prev_num); }

  unsigned int prev_num;
};

// Set number of intraop threads for a local region
struct IntraOpThread
{
  IntraOpThread(unsigned int num)
    : prev_num(at::get_num_threads())
  {
    if (num > 0)
      at::set_num_threads(num);
  }

  ~IntraOpThread() { at::set_num_threads(prev_num); }

  unsigned int prev_num;
};

// Global switch for requesting AD for variable derivatives
struct RequestingAD
{
  RequestingAD() { currently_requesting_AD() = true; }
  ~RequestingAD() { currently_requesting_AD() = false; }
};
}
