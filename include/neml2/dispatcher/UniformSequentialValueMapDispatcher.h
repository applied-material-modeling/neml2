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

#include "neml2/dispatcher/UniformSequentialDispatcher.h"
#include "neml2/dispatcher/ValueMapLoader.h"

namespace neml2
{
/// A simple sequential work dispatcher for dispatching ValueMap using a ValueMapLoader without preprocessing or postprocessing
class UniformSequentialValueMapDispatcher
  : public UniformSequentialDispatcher<ValueMap, ValueMap, ValueMap>
{
public:
  UniformSequentialValueMapDispatcher(Size batch_dim,
                                      std::size_t batch_size,
                                      std::function<ValueMap(ValueMap &&)> && dispatch,
                                      std::function<ValueMap(std::vector<ValueMap> &&)> && reduce =
                                          std::bind(&default_reduce, std::placeholders::_1, 0));

  ValueMap operator()(const ValueMap & x);

protected:
  /// Default reduce function
  static ValueMap default_reduce(std::vector<ValueMap> && results, Size batch_dim);

private:
  const Size _batch_dim;
};
} // namespace neml2
