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
#include "neml2/base/LabeledAxisAccessor.h"

namespace neml2
{
struct AxisLayoutView;

struct AxisLayout
{
  /**
   * @brief Construct a new Axis Layout object
   *
   * @param vars ID-to-variable mapping, partitioned by variable groups
   * @param intmd_shapes ID-to-variable intermediate shape mapping
   * @param base_shapes ID-to-variable base shape mapping
   */
  AxisLayout(const std::vector<std::vector<LabeledAxisAccessor>> &,
             std::vector<TensorShape>,
             std::vector<TensorShape>);

  /// Number of variables
  std::size_t size() const;
  /// Number of variable groups
  std::size_t ngroup() const;
  /// Contiguous view of the variable group
  AxisLayoutView group(std::size_t) const;

  /// ID-to-variable mapping
  std::vector<LabeledAxisAccessor> vars;
  /// ID-to-variable intermediate shape mapping
  std::vector<TensorShape> intmd_shapes;
  /// ID-to-variable base shape mapping
  std::vector<TensorShape> base_shapes;
  /// Offset of each variable group in the layout
  std::vector<std::size_t> offsets;
};

struct AxisLayoutView
{
  /// Number of variables
  std::size_t size() const;
  /// Storage sizes of each variable
  std::vector<Size> storage_sizes(bool include_intmd) const;
  /// ID-to-variable mapping
  ArrayRef<LabeledAxisAccessor> vars;
  /// ID-to-variable intermediate shape mapping
  ArrayRef<TensorShape> intmd_shapes;
  /// ID-to-variable base shape mapping
  ArrayRef<TensorShape> base_shapes;
};
} // namespace neml2
