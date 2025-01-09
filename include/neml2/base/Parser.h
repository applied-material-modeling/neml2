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

} // namespace neml2
