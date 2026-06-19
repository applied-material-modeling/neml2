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

#include <cstddef>
#include <string>

#include "neml2/csrc/dispatchers/WorkScheduler.h"

namespace neml2::aoti
{
/**
 * @brief Dispatch the whole workload to a single device, chunked.
 *
 * The simplest scheduler: every batch chunk goes to one device. Use it to
 * process a batch that does not fit in device memory in fixed-size pieces, to
 * empirically tune the per-call batch size, or as the per-rank scheduler when a
 * host pins one device per process by hand.
 */
class AOTI_EXPORT SimpleScheduler : public SyncScheduler
{
public:
  struct Config
  {
    /// Torch device string, e.g. "cpu", "cuda", or "cuda:1".
    std::string device = "cpu";
    /// Chunk size along the leading batch axis; 0 runs the whole batch at once.
    std::size_t batch_size = 0;
  };

  explicit SimpleScheduler(const Config & config);

  at::Device device() const override { return _device; }
  std::size_t batch_size() const override { return _batch_size; }

private:
  at::Device _device;
  std::size_t _batch_size;
};
} // namespace neml2::aoti
