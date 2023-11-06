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

#include "neml2/models/Data.h"

namespace neml2
{
OptionSet
Data::expected_options()
{
  auto options = NEML2Object::expected_options();
  return options;
}

Data::Data(const OptionSet & options)
  : NEML2Object(options),
    BufferStore(options)
{
}

void
Data::to(const torch::Device & device)
{
  send_buffers_to(device);

  for (auto & data : _registered_data)
    data->to(device);
}

std::map<std::string, BatchTensor>
Data::named_buffers(bool recurse) const
{
  auto buffers = BufferStore::named_buffers();

  if (recurse)
    for (auto & data : _registered_data)
      for (auto && [n, v] : data->named_buffers(true))
        buffers.emplace(data->name() + "." + n, v);

  return buffers;
}

void
Data::register_data(std::shared_ptr<Data> data)
{
  _registered_data.push_back(data.get());
}
}