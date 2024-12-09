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

#include "neml2/misc/error.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <set>

namespace neml2
{
// Forward decl
class EnumSelection;
std::ostream & operator<<(std::ostream &, const EnumSelection &);
std::stringstream & operator>>(std::stringstream &, EnumSelection &);

/**
 * Our poor man's implementation of enum reflection. This is a necessary ingredient for bridging
 * runtime string parsed from input files and static enum types.
 *
 * For developers, this class shall only be used for parsing purposes during the setup phase. Avoid
 * directly working with this class at model evaluation phase at all cost!
 */
class EnumSelection
{
public:
  EnumSelection() = default;
  EnumSelection(const EnumSelection & other);
  EnumSelection(const std::vector<std::string> & candidates, const std::string & selection);
  EnumSelection(const std::vector<std::string> & candidates,
                const std::vector<int> & values,
                const std::string & selection);

  /// Assignment operator
  EnumSelection & operator=(const EnumSelection & other);

  /**
   * The input stream operator shall be the only entry point to modify the selection other than the
   * assignment operator
   */
  friend std::stringstream & operator>>(std::stringstream & ss, EnumSelection &);

  /// Test for equality
  bool operator==(const EnumSelection & other) const;

  /// Test for inequality
  bool operator!=(const EnumSelection & other) const;

  /// Test for selection equality
  bool operator==(const std::string & other) const;

  /// Test for selection inequality
  bool operator!=(const std::string & other) const;

  /// Poor man's reflection implementation
  operator std::string() const { return _selection; }

  /// Implicit conversion to int to let it behave more like a enum
  operator int() const { return _value; }

  /// Candidates
  const std::unordered_map<std::string, int> & candidates() const { return _values; }

  /// Stringified candidates
  std::string candidates_str() const;

  /// Statically cast the enum value to a C++ enum class
  template <typename T>
  T as() const
  {
    return static_cast<T>(_value);
  }

private:
  /// Mapping enum options to int
  std::unordered_map<std::string, int> _values;

  /// Current selection
  std::string _selection;

  /// Current selection's integral value
  int _value;
};

} // namespace neml2
