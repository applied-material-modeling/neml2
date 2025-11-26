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

#include "neml2/base/EnumSelectionBase.h"

namespace neml2
{
// Forward decl
class MultiEnumSelection;
std::ostream & operator<<(std::ostream &, const MultiEnumSelection &);
std::stringstream & operator>>(std::stringstream &, MultiEnumSelection &);

/**
 * @brief Selection of _multiple_ enum value from a list of choices
 * @see neml2::EnumSelectionBase
 */
class MultiEnumSelection : public EnumSelectionBase
{
public:
  MultiEnumSelection() = default;

  /// Create a multi-enum selection from a list of choices and default selections
  MultiEnumSelection(const std::vector<std::string> & choices,
                     const std::vector<std::string> & selections);

  /// Create a multi-enum selection from a list of choices, their corresponding values, and default selections
  MultiEnumSelection(const std::vector<std::string> & choices,
                     const std::vector<int> & values,
                     const std::vector<std::string> & selections);

  /// Select new values
  /// @note This will clear the current selection
  void select(const std::vector<std::string> & selections);

  /// Number of selected values
  std::size_t size() const;

  /// Test for inequality
  bool operator==(const MultiEnumSelection & other) const;

  /// Test for inequality
  bool operator!=(const MultiEnumSelection & other) const;

  /// Poor man's reflection implementation
  const std::vector<std::string> & selections() const { return _selections; }

  /// Selected values cast to int
  const std::vector<int> & values() const { return _values; }

  /// Statically cast the enum value to a C++ enum class
  template <typename T>
  std::vector<T> as() const
  {
    std::vector<T> ret;
    ret.reserve(_values.size());
    for (const auto & v : _values)
      ret.push_back(static_cast<T>(v));
    return ret;
  }

private:
  /// Current selection
  std::vector<std::string> _selections;

  /// Current selection's integral value
  std::vector<int> _values;
};
} // namespace neml2
