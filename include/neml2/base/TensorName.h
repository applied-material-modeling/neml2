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

#include "neml2/misc/types.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
class Model;

/**
 * @brief The name of a tensor object that can be referenced in the input files.
 *
 * All tokens in the input files are essentially strings, and it is not always possible to represent
 * all quantities as strings. This class enables cross-referencing tensor or model output variable
 * (by name) defined in the input file. The object _name_ is used as the label, and the object value
 * is not resolved at parse time. The method TensorName::resolve() is used to explicitly resolve the
 * name to a tensor object or a model's output variable. Note that the resolution takes place upon
 * the first call to TensorName::resolve(), and subsequent calls simply return the cached value.
 */
template <typename T>
struct TensorName
{
public:
  TensorName() = default;

  TensorName(std::string raw)
    : _raw_str(std::move(raw))
  {
  }

  /**
   * @brief Assignment operator
   *
   * This simply assigns the string without parsing and resolving the cross-reference
   */
  TensorName & operator=(const std::string & other)
  {
    _raw_str = other;
    return *this;
  }

  /**
   * @brief Resolve the TensorName to a Tensor object
   *
   * The underlying string is parsed and used to resolve the cross-reference. It is assumed that the
   * cross-referenced tensor object has already been manufactured at this point.
   */
  const T & resolve() const;
  const T & resolve(Model * caller, const std::string & pname) const;

  /// Test equality
  bool operator==(const TensorName & other) const { return _raw_str == other.raw(); }

  /**
   * @brief Get the raw string literal
   *
   * @return const std::string& The raw string literal.
   */
  std::string & raw() { return _raw_str; }
  const std::string & raw() const { return _raw_str; }

private:
  /// Resolve a plain numeric literal
  T resolve_number(Real val) const;

  /// The raw string literal.
  std::string _raw_str;

  /// The resolved tensor value
  mutable T _value;

  /// The tensor or variable referred to by the name
  mutable const T * _tensor = nullptr;
};

/// Stream into a TensorName (used by Parsers to extract input options)
template <typename T>
std::stringstream &
operator>>(std::stringstream & ss, TensorName<T> & t)
{
  ss >> t.raw();
  return ss;
}

/// Stream out a TensorName (used for printing OptionSet)
template <typename T>
std::ostream &
operator<<(std::ostream & os, const TensorName<T> & t)
{
  os << t.raw();
  return os;
}
} // namespace neml2
