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

#include <map>

#include <torch/csrc/jit/frontend/tracer.h>

#include "neml2/misc/types.h"

namespace neml2
{
// Forward decl
class NEML2Object;
class VariableBase;
class Model;
template <typename T>
struct TensorName;
class TensorValueBase;
template <typename T>
class TensorBase;
class Tensor;

namespace jit
{
using namespace torch::jit;
}

/// Interface for object which can store parameters
class ParameterStore
{
public:
  ParameterStore(Model * object);

  ParameterStore(const ParameterStore &) = delete;
  ParameterStore(ParameterStore &&) = delete;
  ParameterStore & operator=(const ParameterStore &) = delete;
  ParameterStore & operator=(ParameterStore &&) = delete;
  virtual ~ParameterStore() = default;

  ///@{
  /// @returns the buffer storage
  const std::map<std::string, std::unique_ptr<TensorValueBase>> & named_parameters() const
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<ParameterStore *>(this)->named_parameters();
  }
  std::map<std::string, std::unique_ptr<TensorValueBase>> & named_parameters();
  ///}@

  /// Get a read-only reference of a parameter
  const TensorValueBase & get_parameter(const std::string & name) const
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<ParameterStore *>(this)->get_parameter(name);
  }
  /// Get a writable reference of a parameter
  TensorValueBase & get_parameter(const std::string & name);

  /// Set the value for a parameter
  void set_parameter(const std::string &, const Tensor &);
  /// Set values for parameters
  void set_parameters(const std::map<std::string, Tensor> &);

protected:
  /**
   * @brief Send parameters to options
   *
   * @param options The target options
   */
  virtual void send_parameters_to(const TensorOptions & options);

  /**
   * @brief Declare a parameter.
   *
   * Note that all parameters are stored in the host (the object exposed to users). An object may be
   * used multiple times in the host, and the same parameter may be declared multiple times. That is
   * allowed, but only the first call to declare_parameter constructs the parameter value, and
   * subsequent calls only returns a reference to the existing parameter.
   *
   * @tparam T Buffer type. See @ref statically-shaped-tensor for supported types.
   * @param name Buffer name
   * @param rawval Buffer value
   * @return Reference to buffer
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name, const T & rawval);

  /**
   * @brief Declare a parameter.
   *
   * Similar to the previous method, but additionally handles the resolution of cross-referenced
   * tensor name.
   *
   * @tparam T Parameter type. See @ref statically-shaped-tensor for supported types.
   * @param name Name of the model parameter.
   * @param tensorname The cross-ref'ed "string" that defines the value of the model parameter.
   * @param allow_nonlinear Whether allows variable coupling
   * @return T The value of the registered model parameter.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name,
                              const TensorName<T> & tensorname,
                              bool allow_nonlinear);

  /**
   * @brief Declare a parameter.
   *
   * Similar to the previous methods, but this method takes care of the high-level logic to directly
   * construct a (possibly nonlinear) parameter from the input option.
   *
   * @tparam T Parameter type. See @ref statically-shaped-tensor for supported types.
   * @param name Name of the model parameter.
   * @param input_option_name Name of the input option that defines the value of the model
   * parameter.
   * @param allow_nonlinear Whether allows variable coupling
   * @return T The value of the registered model parameter.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name,
                              const std::string & input_option_name,
                              bool allow_nonlinear = false);

  /// Assign stack to parameters
  void assign_parameter_stack(jit::Stack & stack);

  /// Collect stack from parameters
  jit::Stack collect_parameter_stack() const;

private:
  Model * _object;

  /// The actual storage for all the parameters
  std::map<std::string, std::unique_ptr<TensorValueBase>> _param_values;

  /// Cached intermediate dimensions of parameters
  std::map<std::string, Size> _param_intmd_dims;
};

} // namespace neml2
