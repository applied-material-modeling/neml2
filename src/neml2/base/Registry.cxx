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

#include <dlfcn.h>
#include <fstream>

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

void
Registry::load(const std::filesystem::path & lib)
{
  namespace fs = std::filesystem;

  // Check that the library exists
  neml_assert(fs::exists(lib), "Runtime library file " + lib.string() + " does not exist.");

  // Check that the library file is readable
  std::ifstream try_open(lib.string().c_str(), std::ifstream::in);
  neml_assert(
      !try_open.fail(),
      "Runtime library file " + lib.string() +
          " exists but could not be opened. Check to make sure that you have read permission.");
  try_open.close();

  // Load the library
  void * const lib_handle = dlopen(fs::absolute(lib).c_str(), RTLD_LAZY);
  neml_assert(lib_handle != nullptr,
              "Runtime library file ",
              lib.string(),
              " exists and can be opened, but cannot by dynamically loaded. This generally means "
              "that the loader was unable to load one or more of the dependencies (see otool or "
              "ldd). Error: \n",
              dlerror());
}

const std::map<std::string, NEML2ObjectInfo> &
Registry::info()
{
  return get()._info;
}

const NEML2ObjectInfo &
Registry::info(const std::string & name)
{
  const auto & reg = get();
  neml_assert(
      reg._info.count(name) > 0,
      name,
      " is not a registered object. Did you forget to register it with register_NEML2_object?");
  return reg._info.at(name);
}

void
Registry::add_inner(const std::string & name,
                    const std::string & type,
                    const OptionSet & options,
                    BuildPtr build_ptr)
{
  auto & reg = get();
  neml_assert(reg._info.count(name) == 0,
              "Duplicate registration found. Object of type ",
              name,
              " is being registered multiple times.");
  reg._info[name] = NEML2ObjectInfo{type, options, build_ptr};
}
} // namespace neml2
