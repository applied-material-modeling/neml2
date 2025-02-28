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
#include <iosfwd>
#include <string>

namespace neml2
{
// Reserved subaxis names
inline const std::string STATE = "state";
inline const std::string OLD_STATE = "old_state";
inline const std::string FORCES = "forces";
inline const std::string OLD_FORCES = "old_forces";
inline const std::string RESIDUAL = "residual";
inline const std::string PARAMETERS = "parameters";

// Reserved subaxis names
std::vector<std::string> reserved_subaxis_names();

/**
 * @brief The accessor containing all the information needed to access an item in a `LabeledAxis`.
 *
 * The accessor consists of an arbitrary number of item names. The last item name can be either a
 * variable name or a sub-axis name. All the other item names are considered to be sub-axis names.
 *
 * The LabeledAxisAccessor stores the item labels and does not resolve the actual layout of the
 * item. This way an accessor can be used access the same variable from different tensor layouts.
 *
 * The item names cannot contain whitespace, ".", ",", ";", or "/".
 */
class LabeledAxisAccessor
{
public:
  LabeledAxisAccessor() = default;

  template <typename... S>
  LabeledAxisAccessor(const char * name, S &&... names)
  {
    validate_item_name(name);
    _item_names.push_back(name);

    (validate_item_name(names), ...);
    (_item_names.push_back(names), ...);
  }

  template <typename... S>
  LabeledAxisAccessor(const std::string & name, S &&... names)
  {
    validate_item_name(name);
    _item_names.push_back(name);

    (validate_item_name(names), ...);
    (_item_names.push_back(names), ...);
  }

  template <typename Container,
            typename = typename std::enable_if_t<
                !std::is_convertible_v<Container, std::string> &&
                std::is_convertible_v<typename std::iterator_traits<
                                          decltype(std::declval<Container>().begin())>::value_type,
                                      std::string> &&
                std::is_convertible_v<typename std::iterator_traits<
                                          decltype(std::declval<Container>().end())>::value_type,
                                      std::string>>>
  LabeledAxisAccessor(const Container & c)
  {
    _item_names = std::vector<std::string>(c.begin(), c.end());
    for (const auto & name : _item_names)
      validate_item_name(name);
  }

  using iterator = std::vector<std::string>::iterator;
  using const_iterator = std::vector<std::string>::const_iterator;

  /**
   * @name Iterators
   *
   * Begin and end iterators for the underlying data.
   */
  ///@{
  iterator begin() { return iterator(_item_names.begin()); }
  iterator end() { return iterator(_item_names.end()); }
  const_iterator begin() const { return const_iterator(_item_names.begin()); }
  const_iterator end() const { return const_iterator(_item_names.end()); }
  ///@}

  explicit operator std::vector<std::string>() const;

  const std::vector<std::string> & vec() const { return _item_names; }

  std::string str() const;

  bool empty() const;

  size_t size() const;

  const std::string & operator[](size_t i) const;

  /// Append a suffix to the final item name.
  LabeledAxisAccessor with_suffix(const std::string & suffix) const;

  /// Append another accessor
  LabeledAxisAccessor append(const LabeledAxisAccessor & axis) const;

  /// Prepend another accessor
  LabeledAxisAccessor prepend(const LabeledAxisAccessor & axis) const;

  /// Remove the leading \p n items from the labels.
  LabeledAxisAccessor slice(int64_t n) const;

  /// Extract out the labels from \p n1 to \p n2
  LabeledAxisAccessor slice(int64_t n1, int64_t n2) const;

  /// A combination of slice and prepend
  LabeledAxisAccessor remount(const LabeledAxisAccessor & axis, int64_t n = 1) const;

  /// Check if this accessor begins with another accessor
  bool start_with(const LabeledAxisAccessor & axis) const;

  ///@{
  /// Check if the accessor is on a specific sub-axis
  bool is_state() const;
  bool is_old_state() const;
  bool is_force() const;
  bool is_old_force() const;
  bool is_residual() const;
  bool is_parameter() const;
  ///@}

  /// Returns the "current" counterpart
  LabeledAxisAccessor current() const;

  /// Returns the "old" counterpart
  LabeledAxisAccessor old() const;

private:
  /// Throws if the item name has invalid format
  void validate_item_name(const std::string &) const;

  std::vector<std::string> _item_names;
};

/// Compare for equality between two LabeledAxisAccessor
bool operator==(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b);

/// Compare for equality between two LabeledAxisAccessor
bool operator!=(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b);

/**
 * @brief The (strict) smaller than operator is created so as to use LabeledAxisAccessor in sorted
 * data structures.
 */
bool operator<(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b);

/**
 * @brief Serialize the \p accessor into a string. The format is simply the concatenation of all the
 * item names delimited by "/".
 */
std::ostream & operator<<(std::ostream & os, const LabeledAxisAccessor & accessor);

using VariableName = LabeledAxisAccessor;
using SubaxisName = LabeledAxisAccessor;

} // namespace neml2
