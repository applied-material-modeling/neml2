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

#include "neml2/base/Registry.h"
#include "neml2/base/OptionSet.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
Registry &
Registry::get()
{
  static Registry registry_singleton;
  return registry_singleton;
}

std::map<std::string, OptionSet>
Registry::expected_options()
{
  auto & reg = get();
  return reg._expected_options;
}

OptionSet
Registry::expected_options(const std::string & name)
{
  auto & reg = get();
  neml_assert(
      reg._expected_options.count(name) > 0,
      name,
      " is not a registered object. Did you forget to register it with register_NEML2_object?");
  return reg._expected_options.at(name);
}

std::string
Registry::syntax_type(const std::string & type)
{
  auto & reg = get();
  return reg._syntax_type[type];
}

BuildPtr
Registry::builder(const std::string & name)
{
  auto & reg = get();
  neml_assert(
      reg._objects.count(name) > 0,
      name,
      " is not a registered object. Did you forget to register it with register_NEML2_object?");
  return reg._objects.at(name);
}

void
Registry::add_inner(const std::string & name,
                    const std::string & type,
                    const OptionSet & options,
                    BuildPtr build_ptr)
{
  auto & reg = get();
  neml_assert(reg._expected_options.count(name) == 0 && reg._objects.count(name) == 0,
              "Duplicate registration found. Object named ",
              name,
              " is being registered multiple times.");

  reg._expected_options[name] = options;
  reg._objects[name] = build_ptr;
  reg._syntax_type[name] = type;
}
} // namespace neml2
