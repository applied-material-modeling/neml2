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

#include <ATen/Context.h>

#include "neml2/drivers/Driver.h"
#include "neml2/base/Registry.h"
#include "neml2/base/Factory.h"

namespace neml2
{
Driver &
get_driver(const std::string & dname)
{
  OptionSet extra_opts;
  return Factory::get_object<Driver>("Drivers", dname, extra_opts, /*force_create=*/false);
}

OptionSet
Driver::expected_options()
{
  OptionSet options = NEML2Object::expected_options();
  options.section() = "Drivers";

  options.set<bool>("verbose") = false;
  options.set("verbose").doc() = "Whether to output additional logging information";

  options.set<Size>("random_seed");
  options.set("random_seed").doc() = "Random seed for any random number generation";

  return options;
}

Driver::Driver(const OptionSet & options)
  : NEML2Object(options),
    DiagnosticsInterface(this),
    _verbose(options.get<bool>("verbose"))
{
  if (options.get("random_seed").user_specified())
    at::manual_seed(options.get<Size>("random_seed"));
}
} // namespace neml2
