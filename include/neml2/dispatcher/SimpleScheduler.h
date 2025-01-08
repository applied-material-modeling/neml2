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

#include "neml2/dispatcher/WorkScheduler.h"

namespace neml2
{
/**
 * @brief A very simple scheduler
 *
 * This schedule is simple in the sense that
 * - It dispatches to a single device
 * - It dispatches a fixed batch size
 * - It does not care if the device is available
 */
class SimpleScheduler : public WorkScheduler
{
public:
  SimpleScheduler(torch::Device device, std::size_t batch_size)
    : _device(device),
      _batch_size(batch_size)
  {
  }

  torch::Device next_device() const override { return _device; };

  std::size_t next_batch_size() const override { return _batch_size; };

  bool is_available(torch::Device, std::size_t) const override { return true; }

  void dispatched(torch::Device, std::size_t) override {}

  void completed(torch::Device, std::size_t) override {}

private:
  torch::Device _device;
  std::size_t _batch_size;
};
} // namespace neml2
