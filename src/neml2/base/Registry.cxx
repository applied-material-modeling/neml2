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
#include "neml2/base/Registry.h"
#include "neml2/models/Model.h"

namespace neml2
{
Registry &
Registry::get_registry()
{
  static Registry registry_singleton;
  return registry_singleton;
}

void
Registry::add_inner(const std::string & name, const ParameterSet & params, BuildPtr build_ptr)
{
  auto & reg = get_registry();
  neml_assert(reg._expected_params.count(name) == 0 && reg._objects.count(name) == 0,
              "Duplicate registration found. Object named ",
              name,
              " is being registered multiple times.");

  reg._expected_params[name] = params;
  reg._objects[name] = build_ptr;
}

void
Registry::print(std::ostream & os) const
{
  for (auto & object : _objects)
    os << object.first << std::endl;
}
} // namespace neml2
