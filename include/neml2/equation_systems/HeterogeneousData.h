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

#include <vector>
#include "neml2/misc/types.h"
#include "neml2/tensors/Tensor.h"

namespace neml2
{
/// Base data structure for discontiguous, heterogeneous data (e.g., HVector and HMatrix)
struct HeterogeneousData
{
  enum class Preference : int8_t
  {
    Assembled,
    Disassembled
  };

  HeterogeneousData() = default;

  HeterogeneousData(const HeterogeneousData &) = default;
  HeterogeneousData(HeterogeneousData &&) noexcept = default;
  HeterogeneousData & operator=(const HeterogeneousData &) = default;
  HeterogeneousData & operator=(HeterogeneousData &&) noexcept = default;
  virtual ~HeterogeneousData() = default;

  HeterogeneousData(std::vector<Tensor>, Preference p = Preference::Disassembled);
  HeterogeneousData(Tensor, Preference p = Preference::Assembled);

  /// Whether any of the contained Tensors require gradients
  bool requires_grad() const;
  /// Tensor options
  TensorOptions options() const;
  /// Whether the disassembled data cache exists
  bool disassembled() const;
  /// Whether the assembled data cache exists
  bool assembled() const;

  /// Get the assembled data (perform assembly on cache miss)
  const Tensor & get_assembled() const;
  /// Get the disassembled data (perform disassembly on cache miss)
  const std::vector<Tensor> & get_disassembled() const;

protected:
  /// Assemble the disassembled data into the assembled data cache
  virtual void assemble() const = 0;
  /// Disassemble the assembled data into the disassembled data cache
  virtual void disassemble() const = 0;

  /// Sub-block tensors
  mutable std::vector<Tensor> _disassembled_data;

  /// Cache for the assembled data. The assembled data is contiguous
  // and flat, and is padded with zeros for missing sub-blocks.
  mutable Tensor _assembled_data;

  /// Preference for data storage when performing operations
  Preference _preference = Preference::Disassembled;
};
} // namespace neml2
