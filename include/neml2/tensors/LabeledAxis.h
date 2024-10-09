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

#include <unordered_map>
#include <type_traits>

#include "neml2/misc/types.h"

#include "neml2/tensors/LabeledAxisAccessor.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"

namespace neml2
{
/**
 * @brief A *labeled* axis used to associate layout of a tensor with human-interpretable names.
 *
 * A logically one-dimensional tensor requires one LabeledAxis, two-dimensional tensor requires two
 * LabeledAxis, and so on. See @ref tensor-labeling for a detailed explanation of tensor labeling.
 *
 * All the LabeledAxis modifiers can only be used during the setup stage. Calling any modifiers
 * after the setup stage is forbidden and will result in a runtime error in Debug mode.
 *
 * All the modifiers return the modified LabeledAxis by reference, so modifiers can be chained.
 * For example
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 * LabeledAxis labels = input().clone();
 * labels.add<SR2>("stress").rename("stress", "stress1").remove("stress1");
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
class LabeledAxis
{
public:
  typedef std::unordered_map<std::string, std::pair<Size, Size>> AxisLayout;

  // Custom comparator for sorting assembly indices
  struct AssemblySliceCmp
  {
    bool operator()(const indexing::TensorIndex & a, const indexing::TensorIndex & b) const
    {
      neml_assert(a.is_slice() && b.is_slice(), "Comparator must be used on slices");
      neml_assert(a.slice().step().expect_int() == 1 && b.slice().step().expect_int() == 1,
                  "Slices must have step == 1");
      return a.slice().start().expect_int() < b.slice().start().expect_int();
    }
  };

  /// Empty constructor
  LabeledAxis(const LabeledAxisAccessor & prefix = {});

  /// (Shallow) copy constructor
  LabeledAxis(const LabeledAxis & other);

  /// Add a variable or subaxis
  template <typename T>
  LabeledAxis & add(const LabeledAxisAccessor & accessor)
  {
    // Add an empty subaxis
    if constexpr (std::is_same_v<LabeledAxis, T>)
    {
      if (!accessor.empty() && !has_subaxis(accessor.slice(0, 1)))
      {
        _subaxes.emplace(accessor.vec()[0], std::make_shared<LabeledAxis>(accessor.slice(0, 1)));
        subaxis(accessor.vec()[0]).add<LabeledAxis>(accessor.slice(1));
      }
      return *this;
    }
    else
    {
      // The storage is *flat* -- will need to reshape when we return!
      // All NEML2 primitive data types will have the member const_base_sizes
      auto sz = utils::storage_size(T::const_base_sizes);
      add(accessor, sz);
      return *this;
    }
  }

  /// Add an arbitrary variable using a `LabeledAxisAccessor`
  LabeledAxis & add(const LabeledAxisAccessor & accessor, Size sz);

  /// Clear all internal data
  void clear();

  /**
   * Setup the layout of all items recursively. The layout of each item is contiguous in memory.
   */
  void setup_layout();

  /// Check the existence of reserved subaxes
  ///@{
  bool has_state() const { return _has_state; }
  bool has_old_state() const { return _has_old_state; }
  bool has_forces() const { return _has_forces; }
  bool has_old_forces() const { return _has_old_forces; }
  bool has_residual() const { return _has_residual; }
  bool has_parameters() const { return _has_parameters; }
  ///@}

  /// Number of variables
  std::size_t nvariable(bool recursive = true) const;

  /// Number of subaxes
  std::size_t nsubaxis(bool recursive = false) const;

  /// Does the item exist?
  bool has_item(const LabeledAxisAccessor & name) const
  {
    return has_variable(name) || has_subaxis(name);
  }

  /// Does the variable of a given primitive type exist?
  template <typename T>
  bool has_variable(const LabeledAxisAccessor & var) const
  {
    if (!has_variable(var))
      return false;

    // also check size
    return storage_size(var) == utils::storage_size(T::const_base_sizes);
  }

  /// Check the existence of a variable by its LabeledAxisAccessor
  bool has_variable(const LabeledAxisAccessor & var) const;

  /// Check the existence of a subaxis by its LabeledAxisAccessor
  bool has_subaxis(const LabeledAxisAccessor & s) const;

  /// Get the total storage size of *this* axis or the storage size of an item
  Size storage_size(const LabeledAxisAccessor & name = {}) const;
  Size storage_size(std::size_t i) const;

  /// Get the layout
  const AxisLayout & layout() const { return _layout; }

  /// Get the indices of a specific item by a `LabeledAxisAccessor`
  indexing::TensorIndex indices(const LabeledAxisAccessor & accessor) const;

  /// Get the common indices of two `LabeledAxis`s
  std::vector<std::pair<indexing::TensorIndex, indexing::TensorIndex>>
  common_indices(const LabeledAxis & other, bool recursive = true) const;

  /// Get the variables
  const std::map<std::string, Size> & variables() const { return _variables; }

  /// Get the variable names
  std::vector<LabeledAxisAccessor> variable_names(bool recursive = true, bool sort = true) const;

  /// Get the fully qualified variable names
  std::vector<LabeledAxisAccessor> qualified_variable_names(bool recursive = true,
                                                            bool sort = true) const;

  /// Get the storage sizes on this axis (by variables)
  std::vector<Size> variable_sizes() const;

  /// Get the subaxes
  const std::map<std::string, std::shared_ptr<LabeledAxis>> & subaxes() const { return _subaxes; }

  /// Get subaxes' names
  std::vector<LabeledAxisAccessor> subaxis_names(bool recursive = false, bool sort = true) const;

  /// Get fully qualified subaxes' names
  std::vector<LabeledAxisAccessor> qualified_subaxis_names(bool recursive = false,
                                                           bool sort = true) const;

  /// Get the storage sizes on this axis (by subaxes)
  std::vector<Size> subaxis_sizes() const;

  /// Get a sub-axis
  const LabeledAxis & subaxis(const LabeledAxisAccessor & name) const;

  /// Get a sub-axis
  LabeledAxis & subaxis(const LabeledAxisAccessor & name);

  /// Check to see if two LabeledAxis objects are equivalent
  bool equals(const LabeledAxis & other) const;

  friend std::ostream & operator<<(std::ostream & os, const LabeledAxis & axis);

private:
  void add(LabeledAxis & axis,
           Size sz,
           const LabeledAxisAccessor::const_iterator & cur,
           const LabeledAxisAccessor::const_iterator & end) const;

  /// Helper method to recursively find the storage size of a variable
  Size storage_size(const LabeledAxisAccessor::const_iterator & cur,
                    const LabeledAxisAccessor::const_iterator & end) const;

  /// Helper method to recursively consume the sub-axis names of a `LabeledAxisAccessor` to get the
  /// indices of a variable.
  indexing::TensorIndex indices(Size offset,
                                const LabeledAxisAccessor::const_iterator & cur,
                                const LabeledAxisAccessor::const_iterator & end) const;

  /// Helper method to (recursively) index this axis using another axis
  void indices(const LabeledAxis & other,
               bool recursive,
               bool inclusive,
               std::vector<Size> & idx,
               Size offset) const;

  /// Get the common indices of two `LabeledAxis`s
  void common_indices(const LabeledAxis & other,
                      bool recursive,
                      std::vector<Size> & idxa,
                      std::vector<Size> & idxb,
                      Size offseta,
                      Size offsetb) const;

  /// Sort a set of LabeledAxisAccessors by their indices
  std::vector<LabeledAxisAccessor>
  sort_by_assembly_order(const std::vector<LabeledAxisAccessor> &) const;

  /// Prefix used to generate fully qualified item names
  const LabeledAxisAccessor _prefix;

  /// Variables and their sizes
  std::map<std::string, Size> _variables;

  /// Sub-axes
  // Each sub-axis can contain its own variables and sub-axes
  std::map<std::string, std::shared_ptr<LabeledAxis>> _subaxes;

  /// The layout of this `LabeledAxis`, i.e. the name-to-indexing::TensorIndices map
  // After all the `LabeledAxis`s are setup, we need to setup the layout once and only once. This is
  // important for performance considerations, as we need to use the layout to construct many,
  // many LabeledVector and LabeledMatrix at runtime, and so we don't want to waste time on setting
  // up the layout over and over again.
  AxisLayout _layout;

  /// The total storage size of the axis.
  // Similar considerations as `_layout`, i.e., the _offset will be zero during the setup stage,
  // and will have a fixed (hopefully correct) size after the layout have been setup.
  Size _offset;

  /// Flags for reserved subaxes
  ///@{
  bool _has_state;
  bool _has_old_state;
  bool _has_forces;
  bool _has_old_forces;
  bool _has_residual;
  bool _has_parameters;
  ///@}
};

std::ostream & operator<<(std::ostream & os, const LabeledAxis & axis);

bool operator==(const LabeledAxis & a, const LabeledAxis & b);

bool operator!=(const LabeledAxis & a, const LabeledAxis & b);
} // namespace neml2
