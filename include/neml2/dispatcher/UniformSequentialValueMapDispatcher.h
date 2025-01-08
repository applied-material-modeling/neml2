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

// #pragma once

// #include "neml2/dispatcher/UniformSequentialDispatcher.h"
// #include "neml2/dispatcher/ValueMapLoader.h"
// #include "neml2/misc/math.h"

// namespace neml2
// {
// /// A simple sequential work dispatcher for dispatching ValueMap using a ValueMapLoader without preprocessing or postprocessing
// template <typename Of = ValueMap, typename Op = ValueMap>
// class UniformSequentialValueMapDispatcher
//   : public UniformSequentialDispatcher<ValueMap, ValueMap, Of, ValueMap, Op>
// {
// public:
//   using BaseDispatcher = UniformSequentialDispatcher<ValueMap, ValueMap, Of, ValueMap, Op>;

//   UniformSequentialValueMapDispatcher(Size batch_dim,
//                                       std::size_t batch_size,
//                                       std::function<ValueMap(ValueMap &&)> && dispatch);

//   ValueMap operator()(const ValueMap & x);

// protected:
//   /// Default reduce function
//   static ValueMap default_reduce(std::vector<ValueMap> && results, Size batch_dim);

// private:
//   const Size _batch_dim;
// };

// template <typename Of, typename Op>
// UniformSequentialValueMapDispatcher<Of, Op>::UniformSequentialValueMapDispatcher(
//     Size batch_dim, std::size_t batch_size, std::function<ValueMap(ValueMap &&)> && dispatch)
//   : BaseDispatcher(batch_size, std::move(dispatch)),
//     _batch_dim(batch_dim)
// {
//   this->_reduce = std::bind(&default_reduce, std::placeholders::_1, batch_dim);
// }

// template <typename Of, typename Op>
// ValueMap
// UniformSequentialValueMapDispatcher<Of, Op>::operator()(const ValueMap & x)
// {
//   ValueMapLoader loader(x, _batch_dim);
//   return this->run(loader);
// }

// template <typename Of, typename Op>
// ValueMap
// UniformSequentialValueMapDispatcher<Of, Op>::default_reduce(std::vector<ValueMap> && results,
//                                                             Size batch_dim)
// {
//   // Re-bin the results
//   std::map<VariableName, std::vector<Tensor>> vars;
//   for (auto && result : results)
//     for (auto && [name, value] : result)
//       vars[name].emplace_back(std::move(value));

//   // Reduce the results
//   ValueMap reduced;
//   for (auto && [name, values] : vars)
//     reduced[name] = math::batch_cat(values, batch_dim);

//   return reduced;
// }
// } // namespace neml2
