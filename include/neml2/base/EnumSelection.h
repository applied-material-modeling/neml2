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
class EnumSelection;
std::ostream & operator<<(std::ostream &, const EnumSelection &);
std::stringstream & operator>>(std::stringstream &, EnumSelection &);

/**
 * @brief Selection of an enum value from a list of choices
 * @see neml2::EnumSelectionBase
 */
class EnumSelection : public EnumSelectionBase
{
public:
  EnumSelection() = default;

  /// Create an enum selection from a list of choices and a default selection
  EnumSelection(const std::vector<std::string> & choices, const std::string & selection);

  /// Create an enum selection from a list of choices, a list of values, and a default selection
  EnumSelection(const std::vector<std::string> & choices,
                const std::vector<int> & values,
                const std::string & selection);

  /// Select a new value
  void select(const std::string & selection);

  /// Test for equality
  bool operator==(const EnumSelection & other) const;

  /// Test for inequality
  bool operator!=(const EnumSelection & other) const;

  /// Test for selection equality
  bool operator==(const std::string & other) const;

  /// Test for selection inequality
  bool operator!=(const std::string & other) const;

  /// Poor man's reflection implementation
  const std::string & selection() const { return _selection; }

  /// Implicit conversion to int to let it behave more like a enum
  int value() const { return _value; }

  /// Statically cast the enum value to a C++ enum class
  template <typename T>
  T as() const
  {
    return static_cast<T>(_value);
  }

private:
  /// Current selection
  std::string _selection = "";

  /// Current selection's integral value
  int _value = 0;
};
} // namespace neml2
