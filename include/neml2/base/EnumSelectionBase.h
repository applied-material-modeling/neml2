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

#include <string>
#include <vector>
#include <unordered_map>
#include <set>

namespace neml2
{
/**
 * Our poor man's implementation of enum reflection. This is a necessary ingredient for bridging
 * runtime string parsed from input files and static enum types.
 *
 * For developers, this class shall only be used for parsing purposes during the setup phase. Avoid
 * directly working with this class at model evaluation phase at all cost!
 */
class EnumSelectionBase
{
public:
  EnumSelectionBase() = default;

  /// Create an enum selection from a list of candidates
  EnumSelectionBase(const std::vector<std::string> & candidates);

  /// Create an enum selection from a list of candidates and their corresponding values
  EnumSelectionBase(const std::vector<std::string> & candidates, const std::vector<int> & values);

  /// Candidates
  const std::unordered_map<std::string, int> & candidates() const { return _candidate_map; }

  /// Stringified candidates
  std::string candidates_str() const;

protected:
  /// Mapping enum options to int
  std::unordered_map<std::string, int> _candidate_map;
};

} // namespace neml2
