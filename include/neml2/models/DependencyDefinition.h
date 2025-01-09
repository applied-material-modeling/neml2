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

#include <set>

namespace neml2
{
/**
 * Defines what this object consume and provide. The consumed and provided items will later
 * be used in DependencyResolver to identify dependencies among a set of objects. In short, this
 * object will _depend_ on whoever provides any of this object's consumed items, and vice versa.
 *
 * @tparam T The type of the consumed/provided items
 */
template <typename T>
class DependencyDefinition
{
public:
  DependencyDefinition() = default;

  DependencyDefinition(DependencyDefinition &&) = delete;
  DependencyDefinition(const DependencyDefinition &) = delete;
  DependencyDefinition & operator=(const DependencyDefinition &) = delete;
  DependencyDefinition & operator=(DependencyDefinition &&) = delete;
  virtual ~DependencyDefinition() = default;

  /// What this object consumes
  virtual std::set<T> consumed_items() const = 0;

  /// What this object provides
  virtual std::set<T> provided_items() const = 0;
};
} // namespace neml2
