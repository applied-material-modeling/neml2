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

#include "neml2/base/NEML2Object.h"
#include "neml2/base/Factory.h"
#include "neml2/base/Settings.h"

namespace neml2
{
OptionSet
NEML2Object::expected_options()
{
  auto options = OptionSet();

  options.set<Factory *>("_factory") = nullptr;
  options.set("_factory").suppressed() = true;

  options.set<std::shared_ptr<Settings>>("_settings") = nullptr;
  options.set("_settings").suppressed() = true;

  options.set<NEML2Object *>("_host") = nullptr;
  options.set("_host").suppressed() = true;

  return options;
}

NEML2Object::NEML2Object(const OptionSet & options)
  : _input_options(options),
    _factory(options.get<Factory *>("_factory")),
    _settings(options.get<std::shared_ptr<Settings>>("_settings")),
    _host(options.get<NEML2Object *>("_host"))
{
}
} // namespace neml2
