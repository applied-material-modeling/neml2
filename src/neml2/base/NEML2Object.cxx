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
#include "neml2/base/TensorName.h"

namespace neml2
{
OptionSet
NEML2Object::expected_options()
{
  auto options = OptionSet();

  options.set<Factory *>("factory") = nullptr;
  options.set("factory").suppressed() = true;

  options.set<std::shared_ptr<Settings>>("settings") = nullptr;
  options.set("settings").suppressed() = true;

  options.set<NEML2Object *>("host") = nullptr;
  options.set("host").suppressed() = true;

  return options;
}

NEML2Object::NEML2Object(const OptionSet & options)
  : _input_options(options),
    _factory(options.get<Factory *>("factory")),
    _settings(options.get<std::shared_ptr<Settings>>("settings")),
    _host(options.get<NEML2Object *>("host"))
{
}

template <typename T>
const T &
NEML2Object::resolve_tensor(const std::string & name)
{
  if (!_input_options.contains(name))
    throw NEMLException("Tensor name '" + name + "' not found in input options of object " +
                        this->name());
  return _input_options.get<TensorName<T>>(name).resolve(_factory);
}

#define NEML2OBJECT_INSTANTIATE(T)                                                                 \
  template const T & NEML2Object::resolve_tensor<T>(const std::string &)
FOR_ALL_TENSORBASE(NEML2OBJECT_INSTANTIATE);
} // namespace neml2
