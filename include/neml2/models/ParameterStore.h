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

#include <ATen/core/stack.h>

#include "neml2/base/NEML2Object.h"
#include "neml2/base/OptionSet.h"
#include "neml2/base/Storage.h"
#include "neml2/tensors/TensorValue.h"

// The following are not directly used by ParameterStore itself.
// We put them here so that derived classes can add expected options of these types.
#include "neml2/base/CrossRef.h"
#include "neml2/base/EnumSelection.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
// Forward decl
class VariableBase;
class Model;

/// Interface for object which can store parameters
class ParameterStore
{
public:
  ParameterStore(OptionSet options, NEML2Object * object);

  ParameterStore(const ParameterStore &) = delete;
  ParameterStore(ParameterStore &&) = delete;
  ParameterStore & operator=(const ParameterStore &) = delete;
  ParameterStore & operator=(ParameterStore &&) = delete;
  virtual ~ParameterStore() = default;

  ///@{
  /// @returns the buffer storage
  const Storage<std::string, TensorValueBase> & named_parameters() const
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<ParameterStore *>(this)->named_parameters();
  }
  Storage<std::string, TensorValueBase> & named_parameters();
  ///}@

  /// Set the value for a parameter
  void set_parameter(const std::string &, const Tensor &);

  /// Set values for parameters
  void set_parameters(const std::map<std::string, Tensor> &);

  /// Get a writable reference of a parameter
  TensorValueBase & get_parameter(const std::string & name);

  /// Get a read-only reference of a parameter
  const TensorValueBase & get_parameter(const std::string & name) const;

  /// Whether this parameter store has any nonlinear parameter
  bool has_nl_param() const { return !_nl_params.empty(); }

  /**
   * @brief Query the existence of a nonlinear parameter
   *
   * @return const VariableBase* Pointer to the VariableBase if the parameter associated with the
   * given parameter name is nonlinear. Returns nullptr otherwise.
   */
  const VariableBase * nl_param(const std::string &) const;

  /// Get all nonlinear parameters
  virtual std::map<std::string, const VariableBase *>
  named_nonlinear_parameters(bool recursive = false) const;

  /// Get all nonlinear parameters' models
  virtual std::map<std::string, Model *>
  named_nonlinear_parameter_models(bool recursive = false) const;

protected:
  /**
   * @brief Send parameters to options
   *
   * @param options The target options
   */
  virtual void send_parameters_to(const torch::TensorOptions & options);

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
   * parameters. Two attempts are made sequentially: first, the method tries to resolve
   * `CrossRef<T>` into `T` directly; if that fails, the method tries to resolve `CrossRef<T>` into
   * a nonlinear parameter where the raw string stored in the cross-ref is treated as the name of
   * the model that defines the nonlinear parameter.
   *
   * @tparam T Parameter type. See @ref statically-shaped-tensor for supported types.
   * @param name Name of the model parameter.
   * @param crossref The cross-ref'ed "string" that defines the value of the model parameter.
   * @param allow_nonlinear Whether allows coupling with a nonlinear parameter
   * @return T The value of the registered model parameter.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T &
  declare_parameter(const std::string & name, const CrossRef<T> & crossref, bool allow_nonlinear);

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
   * @param allow_nonlinear Whether allows coupling with a nonlinear parameter
   * @return T The value of the registered model parameter.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name,
                              const std::string & input_option_name,
                              bool allow_nonlinear = false);

  /// Assign stack to parameters
  void assign_parameter_stack(torch::jit::Stack & stack);

  /// Collect stack from parameters
  torch::jit::Stack collect_parameter_stack() const;

  /// Map from nonlinear parameter names to their corresponding variable views
  std::map<std::string, const VariableBase *> _nl_params;

  /// Map from nonlinear parameter names to models which evaluate them
  std::map<std::string, Model *> _nl_param_models;

private:
  NEML2Object * _object;

  /**
   * @brief Parsed input file options for this object.
   *
   * These options could be convenient when we look up a cross-referenced tensor value by its name.
   *
   */
  const OptionSet _object_options;

  /// The actual storage for all the parameters
  Storage<std::string, TensorValueBase> _param_values;
};

} // namespace neml2
