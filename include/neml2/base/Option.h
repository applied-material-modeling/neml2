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

#include "neml2/base/OptionBase.h"

namespace neml2
{
namespace details
{
/**
 * @name Helper functions for printing
 *
 * Helper functions for printing scalar, vector, vector of vector. Called from
 * Option<T>::print(...).
 *
 * The leading underscore does NOT conform to our naming convention. It is there to avoid ambiguity
 * with customers who try to inject a function with the same name into our namespace, which is
 * unusual but possible with a `using namespace` in the global scope.
 */
///@{
template <typename P>
void _print_helper(std::ostream & os, const P *);
template <typename P>
void _print_helper(std::ostream & os, const std::vector<P> *);
template <typename P>
void _print_helper(std::ostream & os, const std::vector<std::vector<P>> *);
/// The evil vector of bool :/
template <>
void _print_helper(std::ostream & os, const std::vector<bool> *);
/// Specialization so that we don't print out unprintable characters
template <>
void _print_helper(std::ostream & os, const char *);
/// Specialization so that we don't print out unprintable characters
template <>
void _print_helper(std::ostream & os, const unsigned char *);
///@}
}

/**
 * Concrete definition of an option value
 * for a specified type
 */
template <typename T>
class Option : public OptionBase
{
public:
  Option(const std::string & name);

  bool operator==(const OptionBase & other) const override;

  bool operator!=(const OptionBase & other) const override;

  /**
   * \returns A read-only reference to the option value
   */
  const T & get() const { return _value; }

  /**
   * \returns A writable reference to the option value
   */
  T & set() { return _value; }

  void print(std::ostream &) const override;

  std::unique_ptr<OptionBase> clone() const override;

private:
  /// Stored option value
  T _value;
};

namespace details
{
// LCOV_EXCL_START
template <typename P>
void
_print_helper(std::ostream & os, const P * option)
{
  os << *option;
}

template <typename P>
void
_print_helper(std::ostream & os, const std::vector<P> * option)
{
  for (const auto & p : *option)
    os << p << " ";
}

template <typename P>
void
_print_helper(std::ostream & os, const std::vector<std::vector<P>> * option)
{
  for (const auto & pv : *option)
    _print_helper(os, &pv);
}
} // namespace details
// LCOV_EXCL_STOP
} // namespace neml2
