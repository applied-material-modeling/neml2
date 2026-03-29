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
struct AxisLayout
{
  AxisLayout() = default;

  /// Enum for the structure represented by intermediate dimensions (if any)
  enum class IStructure : uint8_t
  {
    DENSE, ///< All intermediate dimensions are grouped into base dimensions
    BLOCK, ///< Intermediate dimensions represent blocks of variables
  };

  /**
   * @brief Construct a new Axis Layout object
   *
   * @param vars ID-to-variable mapping, partitioned by variable groups
   * @param intmd_shapes ID-to-variable intermediate shape mapping
   * @param base_shapes ID-to-variable base shape mapping
   * @param istrs IStructure for each variable group
   */
  AxisLayout(const std::vector<std::vector<LabeledAxisAccessor>> &,
             std::vector<TensorShape>,
             std::vector<TensorShape>,
             std::vector<IStructure>);

  /**
   * @brief Construct a new Axis Layout object by viewing into a parent layout
   *
   * @param parent The axis layout this is viewing into
   * @param start The starting offset of the view (inclusive)
   * @param end The ending offset of the view (exclusive)
   * @param offsets The offsets of the variable groups in the view
   */
  AxisLayout(const AxisLayout * parent,
             std::size_t start,
             std::size_t end,
             std::vector<std::size_t> offsets = {});

  /// Number of variable groups
  std::size_t ngroup() const;
  /// Starting and ending offsets of a variable group
  std::pair<std::size_t, std::size_t> group_offsets(std::size_t) const;
  /// Contiguous view of the variable group
  AxisLayout group(std::size_t) const;
  /// Variable group IStructure
  IStructure group_istr(std::size_t) const;
  /// Contiguous view of the entire layout
  AxisLayout view() const;
  /// Whether this is a view into a parent layout
  bool is_view() const { return _parent != nullptr; }

  /// Number of variables
  std::size_t nvar() const;
  /// Storage sizes of variables
  std::vector<Size> storage_sizes(bool include_intmd) const;

  /// Accessor for variable name
  const LabeledAxisAccessor & var(std::size_t) const;
  /// Accessor for variable intermediate shape
  const TensorShape & intmd_sizes(std::size_t) const;
  /// Accessor for variable base shape
  const TensorShape & base_sizes(std::size_t) const;

private:
  /// ID-to-variable mapping
  std::vector<LabeledAxisAccessor> _vars;
  /// ID-to-variable intermediate shape mapping
  std::vector<TensorShape> _intmd_shapes;
  /// ID-to-variable base shape mapping
  std::vector<TensorShape> _base_shapes;
  /// Offset of each variable group in the layout
  std::vector<std::size_t> _offsets;
  /// IStructure for each variable group
  std::vector<IStructure> _istrs;

  /// The axis layout this is viewing into
  const AxisLayout * _parent = nullptr;
  /// The starting offset of the view (inclusive)
  std::size_t _start = 0;
  /// The ending offset of the view (exclusive)
  std::size_t _end = 0;
};
} // namespace neml2
