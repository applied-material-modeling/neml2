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

#include <map>

#include "neml2/misc/string_utils.h"
#include "neml2/base/LabeledAxisAccessor.h"

namespace neml2
{
class LabeledAxis;

/**
 * @brief A *labeled* axis used to associate layout of a tensor with human-interpretable names.
 *
 * A one-dimensional tensor requires one LabeledAxis, two-dimensional tensor requires two
 * LabeledAxis, and so on. See @ref model-assembly for a detailed explanation of tensor labeling
 * and assembly.
 *
 * Variables and axes can only be added during the setup stage. Adding items after the setup stage
 * is forbidden and will result in a runtime error in Debug mode.
 */
class LabeledAxis
{
public:
  /// Empty constructor
  LabeledAxis() = default;

  /// Sub-axis constructor
  LabeledAxis(LabeledAxisAccessor prefix);

  /// Whether the axis has been set up
  bool is_setup() const { return _setup; }

  /// Return the fully qualified name of an item (i.e. this axis is a sub-axis)
  LabeledAxisAccessor qualify(const LabeledAxisAccessor & accessor) const;

  /// Check the existence of reserved subaxes
  ///@{
  bool has_state() const { return _has_state; }
  bool has_old_state() const { return _has_old_state; }
  bool has_forces() const { return _has_forces; }
  bool has_old_forces() const { return _has_old_forces; }
  bool has_residual() const { return _has_residual; }
  bool has_parameters() const { return _has_parameters; }
  ///@}

  /// Add a sub-axis
  LabeledAxis & add_subaxis(const std::string & name);

  /// Add a variable with known storage size
  void add_variable(const LabeledAxisAccessor & name, Size sz);

  /// Add a variable
  template <typename T>
  void add_variable(const LabeledAxisAccessor & name);

  /// Setup the layout of all items recursively.
  void setup_layout();

  /// Get the storage size of the entire axis
  Size size() const;

  /// Get the storage size of a variable or a local sub-axis
  Size size(const LabeledAxisAccessor & name) const;

  /// Get the slicing indices of a variable or a local sub-axis
  indexing::Slice slice(const LabeledAxisAccessor & name) const;

  /// Get variable information
  ///@{
  /// Number of variables
  std::size_t nvariable() const;
  /// Check the existence of a variable by its name
  bool has_variable(const LabeledAxisAccessor & name) const;
  /// Get the assembly ID of a variable
  std::size_t variable_id(const LabeledAxisAccessor & name) const;
  /// Get the variable names
  const std::vector<LabeledAxisAccessor> & variable_names() const;
  /// Get the variable slicing indices (in assembly order)
  const std::vector<std::pair<Size, Size>> & variable_slices() const;
  /// Get the slicing indices of a variable by name
  const std::pair<Size, Size> & variable_slice(const LabeledAxisAccessor & name) const;
  /// Get the variable storage sizes (in assembly order)
  const std::vector<Size> & variable_sizes() const;
  /// Get the storage size of a variable by name
  Size variable_size(const LabeledAxisAccessor & name) const;
  ///@}

  /// Get sub-axis information
  ///@{
  /// Number of subaxes
  std::size_t nsubaxis() const;
  /// Check the existence of a subaxis by its name
  bool has_subaxis(const LabeledAxisAccessor & name) const;
  /// Get the assembly ID of a sub-axis
  std::size_t subaxis_id(const std::string & name) const;
  /// Get the sub-axes (in assembly order)
  const std::vector<const LabeledAxis *> & subaxes() const;
  /// Get a sub-axis by name
  const LabeledAxis & subaxis(const LabeledAxisAccessor & name) const;
  /// Get a sub-axis by name
  LabeledAxis & subaxis(const LabeledAxisAccessor & name);
  /// Get the sub-axis names
  const std::vector<std::string> & subaxis_names() const;
  /// Get the sub-axis slicing indices (in assembly order)
  const std::vector<std::pair<Size, Size>> & subaxis_slices() const;
  /// Get the slicing indices of a sub-axis by name
  std::pair<Size, Size> subaxis_slice(const LabeledAxisAccessor & name) const;
  /// Get the sub-axis storage sizes (in assembly order)
  const std::vector<Size> & subaxis_sizes() const;
  /// Get the storage size of a sub-axis by name
  Size subaxis_size(const LabeledAxisAccessor & name) const;
  ///@}

  /// Check to see if two axes are equivalent
  bool equals(const LabeledAxis & other) const;

  friend std::ostream & operator<<(std::ostream & os, const LabeledAxis & axis);

private:
  /// Cache the existence of a reserved subaxis
  void cache_reserved_subaxis(const std::string & axis_name);

  /// Ensure that the axis has been setup
  void ensure_setup_dbg() const;

  /// Whether the axis has been setup
  bool _setup = false;

  /// Prefix used to generate fully qualified item names
  const LabeledAxisAccessor _prefix = {};

  /// The total storage size of the axis
  Size _size = 0;

  /// Variables and their sizes
  std::map<std::string, Size> _variables;

  /// Sub-axes
  std::map<std::string, std::shared_ptr<LabeledAxis>> _subaxes;

  /**
   * @brief Variable maps for assembly purposes
   *
   * These maps are set up by neml2::LabeledAxis::setup_layout.
   * These maps include variables from all sub-axes (recursively).
   */
  ///@{
  /// Map from variable names to their assembly ID
  std::map<LabeledAxisAccessor, std::size_t> _variable_to_id_map;
  /// Map from assembly ID to variable names
  std::vector<LabeledAxisAccessor> _id_to_variable_map;
  /// Map from assembly ID to variable storage size
  std::vector<Size> _id_to_variable_size_map;
  /// Map from assembly ID to variable slicing indices
  std::vector<std::pair<Size, Size>> _id_to_variable_slice_map;
  ///@}

  /**
   * @brief Sub-axis maps for assembly purposes
   *
   * These maps are set up by neml2::LabeledAxis::setup_layout.
   * These maps ONLY include local sub-axes.
   */
  ///@{
  /// Map from assembly ID to sub-axes
  std::vector<const LabeledAxis *> _sorted_subaxes;
  /// Map from sub-axis names to their assembly ID
  std::map<std::string, std::size_t> _subaxis_to_id_map;
  /// Map from assembly ID to sub-axis names
  std::vector<std::string> _id_to_subaxis_map;
  /// Map from assembly ID to sub-axis storage size
  std::vector<Size> _id_to_subaxis_size_map;
  /// Map from assembly ID to sub-axis slicing indices
  std::vector<std::pair<Size, Size>> _id_to_subaxis_slice_map;
  ///@}

  /// Flags for reserved subaxes
  ///@{
  bool _has_state = false;
  bool _has_old_state = false;
  bool _has_forces = false;
  bool _has_old_forces = false;
  bool _has_residual = false;
  bool _has_parameters = false;
  ///@}
};

std::ostream & operator<<(std::ostream & os, const LabeledAxis & axis);

bool operator==(const LabeledAxis & a, const LabeledAxis & b);

bool operator!=(const LabeledAxis & a, const LabeledAxis & b);
} // namespace neml2
