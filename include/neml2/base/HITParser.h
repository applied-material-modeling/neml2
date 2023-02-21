// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/base/Parser.h"
#include "hit.h"
#include <memory>

namespace neml2
{
/// A helper class to deserialize a file written in HIT format
class HITParser : public Parser
{
public:
  HITParser() = default;

  virtual void parse(const std::string & filename);

  /// Get the root of the parsed input file
  hit::Node * root() { return _root.get(); }

  /// Extract (and cast) parameters into the parameter collection
  virtual ParameterCollection parameters() const;

private:
  class ExtractParamsWalker : public hit::Walker
  {
  public:
    ExtractParamsWalker(ParameterSet & params)
      : _params(params)
    {
    }

    void walk(const std::string & fullpath, const std::string & nodepath, hit::Node * n) override;

  private:
    ParameterSet & _params;
  };

  /// The root node of the parsed input file
  std::unique_ptr<hit::Node> _root;
};

} // namespace neml2
