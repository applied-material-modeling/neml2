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
#include "neml2/tensors/tensors_fwd.h"

namespace neml2
{
/**
 * @brief The name of a tensor object that can be referenced in the input files.
 *
 * All tokens in the input files are essentially strings, and it is not always possible to represent
 * all quantities as strings. This enables cross-referencing tensor objects. The object _name_ is
 * used as the label, and the object value is not resolved at runtime.
 */
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
  TensorName & operator=(const std::string & other);

  /**
   * @brief Explicit conversion operator.
   *
   * The underlying string is parsed and used to resolve the cross-reference. It is assumed that the
   * cross-referenced tensor object has already been manufactured at this point.
   */
  ///@{
  explicit operator ATensor() const;
  explicit operator Tensor() const;
#define DECL_CONVERSION(T) explicit operator T() const
  FOR_ALL_PRIMITIVETENSOR(DECL_CONVERSION);
#undef DECL_CONVERSION
  ///@}

  /// Test equality
  bool operator==(const TensorName & other) const { return _raw_str == other.raw(); }

  /**
   * @brief Get the raw string literal
   *
   * @return const std::string& The raw string literal.
   */
  const std::string & raw() const { return _raw_str; }

  friend std::stringstream & operator>>(std::stringstream & in, TensorName &);

private:
  /// The raw string literal.
  std::string _raw_str;
};

/// Stream into a TensorName (used by Parsers to extract input options)
std::stringstream & operator>>(std::stringstream &, TensorName &);

/// Stream out a TensorName (used for printing OptionSet)
std::ostream & operator<<(std::ostream & os, const TensorName &);
} // namespace neml2
