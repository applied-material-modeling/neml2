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

#include "neml2/base/NEML2Object.h"
#include "neml2/models/BufferStore.h"
#include "neml2/base/Factory.h"

namespace neml2
{
class Data : public NEML2Object, public BufferStore
{
public:
  static OptionSet expected_options();

  /**
   * @brief Construct a new Data object
   *
   * @param options The set of options extracted from the input file
   */
  Data(const OptionSet & options);

  /// All the registered data objects
  const std::vector<Data *> & registered_data() const { return _registered_data; }

protected:
  /**
   * Register a data object and return its reference
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Data, T>>>
  T & register_data(const std::string & name)
  {
    OptionSet extra_opts;
    extra_opts.set<NEML2Object *>("_host") = host();
    auto data = Factory::get_object_ptr<T>("Data", name, extra_opts, /*force_create=*/false);

    if (std::find(_registered_data.begin(), _registered_data.end(), data.get()) !=
        _registered_data.end())
      throw SetupException("Data named '" + name + "' has already been registered.");

    _registered_data.push_back(data.get());
    return *data;
  }

  /// Registered Data objects
  std::vector<Data *> _registered_data;
};
} // namespace neml2
