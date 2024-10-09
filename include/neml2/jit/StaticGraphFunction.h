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

#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/api/function_impl.h>

#include "neml2/jit/utils.h"
#include "neml2/misc/error.h"

namespace neml2
{
namespace jit
{
/**
 * @brief A traced function with static call signature (input and output types determined at
 * compile-time).
 *
 * @tparam OutputType Output type
 * @tparam InputTypes Input types
 */
template <typename OutputType, typename... InputTypes>
class StaticGraphFunction
{
public:
  /**
   * @brief Construct a new StaticGraphFunction object by tracing the given function with the
   given
   * example inputs
   *
   * @param name Function name
   * @param f Function to trace
   * @param x Example inputs
   */
  StaticGraphFunction(c10::QualifiedName name,
                      const std::function<OutputType(InputTypes &...)> & f,
                      const std::tuple<InputTypes...> & x);

  /// Call the traced function and return the outputs
  OutputType operator()(const InputTypes &... args);

  /// Return the underlying graph function
  const torch::jit::GraphFunction & function() const { return _graph_function; }

  /// Return the underlying graph function
  torch::jit::GraphFunction & function() { return _graph_function; }

private:
  /// Helper to create the traced function graph
  std::shared_ptr<torch::jit::Graph>
  create_graph(const std::function<OutputType(InputTypes &...)> & f,
               const std::tuple<InputTypes...> & x) const;

  /// The function responsible for handling the function call
  torch::jit::GraphFunction _graph_function;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename OutputType, typename... InputTypes>
StaticGraphFunction<OutputType, InputTypes...>::StaticGraphFunction(
    c10::QualifiedName name,
    const std::function<OutputType(InputTypes &...)> & f,
    const std::tuple<InputTypes...> & x)
  : _graph_function(name, create_graph(f, x), nullptr)
{
}

template <typename OutputType, typename... InputTypes>
OutputType
StaticGraphFunction<OutputType, InputTypes...>::operator()(const InputTypes &... args)
{
  auto stack = tuple_to_stack(std::make_tuple(args...));
  _graph_function.run(stack);
  neml_assert_dbg(stack.size() == std::tuple_size_v<OutputType>,
                  "Number of outputs on the stack does not match the expected number of outputs");
  OutputType output;
  details::stack_to_tuple_impl<std::tuple_size_v<OutputType> - 1>(stack, output);
  return output;
}

template <typename OutputType, typename... InputTypes>
std::shared_ptr<torch::jit::Graph>
StaticGraphFunction<OutputType, InputTypes...>::create_graph(
    const std::function<OutputType(InputTypes &...)> & f, const std::tuple<InputTypes...> & x) const
{
  auto var_name_lookup_fn = [](const torch::autograd::Variable & /*var*/) -> std::string
  { return ""; };
  auto f_wrap = [&f](torch::jit::Stack inputs) -> torch::jit::Stack
  {
    auto x = stack_to_tuple<InputTypes...>(inputs);
    return tuple_to_stack(std::apply(f, x));
  };
  return std::move(std::get<0>(torch::jit::tracer::trace(tuple_to_stack(x),
                                                         f_wrap,
                                                         var_name_lookup_fn,
                                                         /*strict=*/true,
                                                         /*force_outplace=*/false))
                       ->graph);
}

} // namespace jit
} // namespace neml2
