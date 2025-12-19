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

#include "neml2/neml2.h"
#include "neml2/base/HITParser.h"

namespace neml2
{
std::unique_ptr<Factory>
load_input(const std::filesystem::path & path, const std::string & additional_input)
{
  // Force link dynamic libraries
  ::_neml2_force_link_models();
  ::_neml2_force_link_solvers();
  ::_neml2_force_link_user_tensors();
  ::_neml2_force_link_drivers();

  // For now we only support HIT
  if (utils::end_with(path.string(), ".i"))
  {
    HITParser parser;
    auto inp = parser.parse(path, additional_input);
    return std::make_unique<Factory>(inp);
  }
  else
    throw ParserException("Unsupported parser type");
}

std::shared_ptr<Model>
load_model(const std::filesystem::path & path, const std::string & mname)
{
  auto factory = load_input(path);
  return factory->get_model(mname);
}
} // namespace neml2
