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

#include <filesystem>

#include "neml2/misc/string_utils.h"

namespace neml2
{
class OptionCollection;
class LabeledAxisAccessor;
using VariableName = LabeledAxisAccessor;
class EnumSelection;
class MultiEnumSelection;

/**
 * @brief A parser is responsible for parsing an input file into a collection of options which
 * can be used by the Factory to manufacture corresponding objects.
 *
 */
class Parser
{
public:
  Parser() = default;

  Parser(const Parser &) = default;
  Parser(Parser &&) noexcept = default;
  Parser & operator=(const Parser &) = default;
  Parser & operator=(Parser &&) noexcept = default;
  virtual ~Parser() = default;

  /// Known top-level sections in the input file
  static const std::vector<std::string> sections;

  /**
   * @brief Deserialize a file.
   *
   * @param filename Name/path of the input file.
   * @param additional_input  Additional content of the input file not included in the input file
   * itself, e.g., from command line.
   * @return OptionCollection The extracted object options.
   */
  virtual OptionCollection parse(const std::filesystem::path & filename,
                                 const std::string & additional_input = "") const = 0;
};

namespace utils
{
template <typename T>
void
parse_(T & val, const std::string & raw_str)
{
  std::stringstream ss(trim(raw_str));
  ss >> val;
  if (ss.fail())
    throw ParserException("Failed to parse '" + raw_str + "' as a " +
                          utils::demangle(typeid(T).name()));
}

template <typename T>
T
parse(const std::string & raw_str)
{
  T val;
  parse_(val, raw_str);
  return val;
}

template <typename T>
void
parse_vector_(std::vector<T> & vals, const std::string & raw_str)
{
  auto tokens = split(raw_str, " \t\n\v\f\r");
  vals.resize(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++)
    parse_<T>(vals[i], tokens[i]);
}

template <typename T>
std::vector<T>
parse_vector(const std::string & raw_str)
{
  std::vector<T> vals;
  parse_vector_(vals, raw_str);
  return vals;
}

template <typename T>
void
parse_vector_vector_(std::vector<std::vector<T>> & vals, const std::string & raw_str)
{
  auto token_vecs = split(raw_str, ";");
  vals.resize(token_vecs.size());
  for (size_t i = 0; i < token_vecs.size(); i++)
    parse_vector_<T>(vals[i], token_vecs[i]);
}

template <typename T>
std::vector<std::vector<T>>
parse_vector_vector(const std::string & raw_str)
{
  std::vector<std::vector<T>> vals;
  parse_vector_vector_(vals, raw_str);
  return vals;
}

// template specializations for special option types
template <>
void parse_<bool>(bool &, const std::string & raw_str);
/// This special one is for the evil std::vector<bool>!
template <>
void parse_vector_<bool>(std::vector<bool> &, const std::string & raw_str);
template <>
void parse_<TensorShape>(TensorShape &, const std::string & raw_str);
template <>
void parse_<VariableName>(VariableName &, const std::string & raw_str);
template <>
void parse_<Device>(Device &, const std::string & raw_str);
} // namespace utils
} // namespace neml2
