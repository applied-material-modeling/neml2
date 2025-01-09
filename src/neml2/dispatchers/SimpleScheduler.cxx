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

#include "neml2/dispatchers/SimpleScheduler.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
SimpleScheduler::SimpleScheduler(Device device, std::size_t batch_size, std::size_t capacity)
  : _device(device),
    _batch_size(batch_size),
    _capacity(capacity)
{
}

bool
SimpleScheduler::schedule_work(Device & device, std::size_t & batch_size) const
{
  if (_load + _batch_size > _capacity)
    return false;

  device = _device;
  batch_size = _batch_size;
  return true;
}

void
SimpleScheduler::dispatched_work(Device, std::size_t n)
{
  _load += n;
}

void
SimpleScheduler::completed_work(Device, std::size_t n)
{
  neml_assert(_load >= n, "Load underflow");
  _load -= n;
}
} // namespace neml2
