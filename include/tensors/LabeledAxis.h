#pragma once

#include <unordered_map>
#include <type_traits>

#include "misc/types.h"

#include "tensors/Scalar.h"
#include "tensors/SymR2.h"

namespace neml2
{
/// The list of all NEML2 primitive data types that can be labeled
template <typename T>
struct is_labelable : std::false_type
{
};
template <>
struct is_labelable<Scalar> : std::true_type
{
};
template <>
struct is_labelable<SymR2> : std::true_type
{
};

// Forward declarations
class LabeledAxis;

typedef std::unordered_map<std::string, TorchIndex> AxisLayout;

/// The LabeledAxis contains all information regarding how an axis of a LabeledTensor is labeled.
/// **This class doesn't actually store data.**
///
/// Some naming conventions:
///   - Item: A labelable chunk of data
///   - Variable: An item that is also of a NEML2 primitive data type
///   - Sub-axis: An item of type LabeledAxis
///
/// So yes, an axis can be labeled recursively, e.g.
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///  0 1 2 3 4 5     6     7 8 9 10 11 12   13   14
/// |-----------| |-----| |              | |  | |  |
///    sub a       sub b  |              | |  | |  |
/// |-------------------| |--------------| |--| |--|
///          sub                  a          b    c
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// The above example represents an axis of size 15.
/// This LabeledAxis has 4 items: `a`, `b`, `c`, `sub`
/// `a` is a variable of type `SymR2`, `b` and `c` are variables of type `Scalar`, and
/// `sub` is a sub-axis of type `LabeledAxis`.
/// `sub` by itself represents an axis of size 7, containing 2 items:
/// `sub a` is a variable of type `SymR2` and `sub b` is a variable of type `Scalar`.
class LabeledAxis
{
public:
  LabeledAxis();

  /// (Deep) copy constructor
  LabeledAxis(const LabeledAxis & other);

  /**
   * All the LabeledAxis modifiers. The modifiers can only be used during the setup stage. Calling
   * any modifiers after the setup stage is forbidden and will result in a runtime error in Debug
   * mode.
   *
   * All the modifiers return the modified LabeledAxis by reference, so modifiers can be chained.
   For
   * example
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   LabeledAxis labels = input().clone();
   labels.add<SymR2>("stress").rename("stress", "stress1").remove("stress1");
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   */
  /// @{
  /// Add a variable or subaxis
  template <typename T>
  LabeledAxis & add(const std::string & name)
  {
    // The storage is *flat* -- will need to reshape when we return!
    // All NEML2 primitive data types will have the member _base_sizes
    if constexpr (is_labelable<T>::value)
    {
      auto sz = utils::storage_size(T::_base_sizes);
      add(name, sz);
      return *this;
    }

    // Add an empty subaxis
    if constexpr (std::is_same_v<LabeledAxis, T>)
    {
      if (!has_subaxis(name))
        _subaxes.emplace(name, std::make_shared<LabeledAxis>());
      return *this;
    }
  }

  /// Add an arbitrary variable with known storage size.
  LabeledAxis & add(const std::string & name, TorchSize sz);

  /// Merge with another `LabeledAxis`. Note that the prefixes and suffixes of the merged axis won't carry over.
  LabeledAxis & merge(LabeledAxis & other);

  /// Add prefix to the labels.
  /// Note that this method doesn't recurse through sub-axes.
  LabeledAxis & prefix(const std::string & prefix, const std::string & delimiter = "_");

  /// Add suffix to the labels.
  /// Note that this method doesn't recurse through sub-axes.
  LabeledAxis & suffix(const std::string & suffix, const std::string & delimiter = "_");

  /// Change the label of an item
  LabeledAxis & rename(const std::string & original, const std::string & rename);

  /// Remove an item
  LabeledAxis & remove(const std::string & name);

  /// Clear everything
  LabeledAxis & clear();
  /// @}

  /**
  Setup the layout of all items recursively. The layout of each item is contiguous in memory.

  NOTE: Since we are using `std::unordered_map` to store the variables and subaxes, it is possible
  that two logically same `LabeledAxis`s result in different layouts. Let's try to avoid that
  by sorting the variables and subaxes before generating the layout. Yes, the sorting will slow
  things down. But since we make sure all setup_layout() calls happen at the model construction
  phase, it is okay to be slow :)
  */
  void setup_layout();

  /// Number of items
  size_t nitem() const { return nvariable() + nsubaxis(); }

  /// Number of variables
  size_t nvariable() const { return _variables.size(); }

  /// Number of subaxes
  size_t nsubaxis() const { return _subaxes.size(); }

  /// Does the item exist?
  bool has_item(const std::string & name) const { return has_variable(name) || has_subaxis(name); }

  /// Does the variable of a given primitive type exist?
  template <typename T>
  bool has_variable(const std::string & name) const
  {
    if constexpr (is_labelable<T>::value)
    {
      if (!has_variable(name))
        return false;

      // also check size
      auto sz = utils::storage_size(T::_base_sizes);
      return _variables.at(name) == sz;
    }
  }

  /// Does the variable exist?
  bool has_variable(const std::string & name) const { return _variables.count(name); }

  /// Does the item exist?
  bool has_subaxis(const std::string & name) const { return _subaxes.count(name); }

  /// (total) storage size
  /// @{
  TorchSize storage_size() const { return _offset; }
  TorchSize storage_size(const std::string &) const;
  /// @}

  /// Get the layout
  const AxisLayout & layout() const { return _layout; }

  /// Get the indices of a specific item
  const TorchIndex & indices(const std::string & name) const;

  /// Get the item names
  std::vector<std::string> item_names() const;

  /// Get the variables
  const std::unordered_map<std::string, TorchSize> & variables() const { return _variables; }

  /// Get the subaxes
  const std::unordered_map<std::string, std::shared_ptr<LabeledAxis>> & subaxes() const
  {
    return _subaxes;
  }

  /// Get a sub-axis
  /// @{
  const LabeledAxis & subaxis(const std::string & name) const;
  LabeledAxis & subaxis(const std::string & name);
  /// @}

  /// Check to see if two LabeledAxis objects are equivalent
  bool equals(const LabeledAxis & other) const;

  friend std::ostream & operator<<(std::ostream & os, const LabeledAxis & info);

  /// Write this object in dot format
  void to_dot(std::ostream & os,
              int & id,
              std::string name = "",
              bool subgraph = false,
              bool node_handle = false) const;

  /// Helper static variable to keep track of indentation when printing
  static int level;

private:
  /// Variables and their sizes
  std::unordered_map<std::string, TorchSize> _variables;

  /// Sub-axes
  // Each sub-axis can contain its own variables and sub-axes
  std::unordered_map<std::string, std::shared_ptr<LabeledAxis>> _subaxes;

  /// The layout of this `LabeledAxis`, i.e. the name-to-TorchSlice map
  // After all the `LabeledAxis`s are setup, we need to setup the layout once and only once. This is
  // important for performance considerations, as we need to use the layout to construct many,
  // many LabeledVector and LabeledMatrix at runtime, and so we don't want to waste time on setting
  // up the layout over and over again.
  AxisLayout _layout;

  /// The total storage size of the axis.
  // Similar considerations as `_layout`, i.e., the _offset will be zero during the setup stage,
  // and will have a fixed (hopefully correct) size after the layout have been setup.
  TorchSize _offset;
};

/// Print LabeledAxis to the ostream
std::ostream & operator<<(std::ostream & os, const LabeledAxis & info);

/// Equality between LabeledAxis objects, expressed via the equals method
bool operator==(const LabeledAxis & a, const LabeledAxis & b);

/// Inequality between LabeledAxis objects
bool operator!=(const LabeledAxis & a, const LabeledAxis & b);
} // namespace neml2
