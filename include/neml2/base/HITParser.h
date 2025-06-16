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

#include "hit/parse.h"
#include "neml2/base/Parser.h"

namespace neml2
{
class OptionSet;

/**
 * @copydoc neml2::Parser
 *
 * The HITParser parses input files written in the [HIT format](https://github.com/idaholab/hit).
 */
class HITParser : public Parser
{
public:
  HITParser() = default;

  HITParser(const HITParser &) = default;
  HITParser(HITParser &&) noexcept = default;
  HITParser & operator=(const HITParser &) = default;
  HITParser & operator=(HITParser &&) noexcept = default;
  ~HITParser() override = default;

  /// Parse a HIT input file from a filename.
  InputFile parse(const std::filesystem::path & filename,
                  const std::string & additional_input = "") const override;

  /// Parse a HIT input file from a root node.
  InputFile parse(hit::Node * root) const;

private:
  /**
   * @brief Extract options for a specific object.
   *
   * @param object The object whose options are to be extracted.
   * @param section The current section node.
   * @return OptionSet The options of the object.
   */
  virtual OptionSet extract_object_options(hit::Node * object, hit::Node * section) const;

  void extract_options(hit::Node * object, OptionSet & options) const;
  void extract_option(hit::Node * node, OptionSet & options) const;
};

} // namespace neml2
