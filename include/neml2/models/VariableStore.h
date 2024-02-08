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

#pragma once

#include "neml2/base/NEML2Object.h"
#include "neml2/base/Storage.h"
#include "neml2/tensors/Variable.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
class VariableStore
{
public:
  VariableStore(const OptionSet & options, NEML2Object * object);

  LabeledAxis & declare_axis(const std::string & name);

  /// Setup the layouts of all the registered axes
  virtual void setup_layout();

  /// Get an input variable
  /// @{
  template <typename T = BatchTensor>
  Variable<T> & get_input_variable(const LabeledAxisAccessor & name)
  {
    auto var_base_ptr = _input_views.query_value(name);
    neml_assert(var_base_ptr, "Input variable ", name, " does not exist.");
    auto var_ptr = dynamic_cast<Variable<T> *>(var_base_ptr);
    neml_assert(
        var_ptr, "Input variable ", name, " exist but cannot be cast to the requested type.");
    return *var_ptr;
  }
  template <typename T = BatchTensor>
  const Variable<T> & get_input_variable(const LabeledAxisAccessor & name) const
  {
    const auto var_base_ptr = _input_views.query_value(name);
    neml_assert(var_base_ptr, "Input variable ", name, " does not exist.");
    const auto var_ptr = dynamic_cast<const Variable<T> *>(var_base_ptr);
    neml_assert(
        var_ptr, "Input variable ", name, " exist but cannot be cast to the requested type.");
    return *var_ptr;
  }
  /// @}

  /// Get an output variable
  /// @{
  template <typename T = BatchTensor>
  const Variable<T> & get_output_variable(const LabeledAxisAccessor & name)
  {
    return std::as_const(*this).get_output_variable<T>(name);
  }
  template <typename T = BatchTensor>
  const Variable<T> & get_output_variable(const LabeledAxisAccessor & name) const
  {
    const auto var_base_ptr = _output_views.query_value(name);
    neml_assert(var_base_ptr, "Output variable ", name, " does not exist.");
    const auto var_ptr = dynamic_cast<const Variable<T> *>(var_base_ptr);
    neml_assert(
        var_ptr, "Output variable ", name, " exist but cannot be cast to the requested type.");
    return *var_ptr;
  }
  /// @}

  /// Definition of the input variables
  /// @{
  LabeledAxis & input_axis() { return _input_axis; }
  const LabeledAxis & input_axis() const { return _input_axis; }
  /// @}

  /// Which variables this object defines as output
  /// @{
  LabeledAxis & output_axis() { return _output_axis; }
  const LabeledAxis & output_axis() const { return _output_axis; }
  /// @}

  /// Input variable views
  /// @{
  Storage<LabeledAxisAccessor, VariableBase> & input_views() { return _input_views; }
  const Storage<LabeledAxisAccessor, VariableBase> & input_views() const { return _input_views; }
  /// @}

  /// Output variable views
  /// @{
  Storage<LabeledAxisAccessor, VariableBase> & output_views() { return _output_views; }
  const Storage<LabeledAxisAccessor, VariableBase> & output_views() const { return _output_views; }
  /// @}

  /// Input storage
  /// @{
  LabeledVector & input_storage() { return _in; }
  const LabeledVector & input_storage() const { return _in; }
  /// @}

  /// Output storage
  /// @{
  LabeledVector & output_storage() { return _out; }
  const LabeledVector & output_storage() const { return _out; }
  /// @}

  /// Derivative storage
  /// @{
  LabeledMatrix & derivative_storage() { return _dout_din; }
  const LabeledMatrix & derivative_storage() const { return _dout_din; }
  /// @}

  /// Second derivative storage
  /// @{
  LabeledTensor3D & second_derivative_storage() { return _d2out_din2; }
  const LabeledTensor3D & second_derivative_storage() const { return _d2out_din2; }
  /// @}

  /// Get the view of an input variable
  VariableBase * input_view(const LabeledAxisAccessor &);
  /// Get the view of an output variable
  VariableBase * output_view(const LabeledAxisAccessor &);

protected:
  /// Cache the variable's batch shape
  virtual void cache(TorchShapeRef batch_shape);

  virtual void allocate_variables(TorchShapeRef batch_shape, const torch::TensorOptions & options);

  virtual void setup_input_views();

  virtual void setup_output_views(bool out, bool dout_din = true, bool d2out_din2 = true);

  virtual void detach_and_zero(bool out, bool dout_din = true, bool d2out_din2 = true);

  /// Declare an input variable
  template <typename T>
  const Variable<T> & declare_input_variable(const LabeledAxisAccessor & name)
  {
    declare_variable<T>(_input_axis, name);
    return *create_variable_view<T>(_input_views, name);
  }

  /// Declare an input variable (with unknown base shape at compile time)
  const Variable<BatchTensor> & declare_input_variable(const LabeledAxisAccessor & name,
                                                       TorchSize sz)
  {
    declare_variable(_input_axis, name, sz);
    return *create_variable_view<BatchTensor>(_input_views, name, sz);
  }

  /// Declare an input variable that is a list of tensors of fixed size
  template <typename T>
  const Variable<BatchTensor> & declare_input_variable_list(const LabeledAxisAccessor & name,
                                                            TorchSize list_size)
  {
    return declare_input_variable(name, list_size * T::const_base_storage);
  }

  /// Declare an output variable
  template <typename T>
  Variable<T> & declare_output_variable(const LabeledAxisAccessor & name)
  {
    declare_variable<T>(_output_axis, name);
    return *create_variable_view<T>(_output_views, name);
  }

  /// Declare an input variable (with unknown base shape at compile time)
  Variable<BatchTensor> & declare_output_variable(const LabeledAxisAccessor & name, TorchSize sz)
  {
    declare_variable(_output_axis, name, sz);
    return *create_variable_view<BatchTensor>(_output_views, name, sz);
  }

  /// Declare an output variable that is a list of tensors of fixed size
  template <typename T>
  Variable<BatchTensor> & declare_output_variable_list(const LabeledAxisAccessor & name,
                                                       TorchSize list_size)
  {
    return declare_output_variable(name, list_size * T::const_base_storage);
  }

  /// Declare an item recursively on an axis
  template <typename T>
  LabeledAxisAccessor declare_variable(LabeledAxis & axis, const LabeledAxisAccessor & var) const
  {
    return declare_variable(axis, var, T::const_base_storage);
  }

  /// Declare an item (with known storage size) recursively on an axis
  LabeledAxisAccessor
  declare_variable(LabeledAxis & axis, const LabeledAxisAccessor & var, TorchSize sz) const
  {
    axis.add(var, sz);
    return var;
  }

  /// Declare a subaxis recursively on an axis
  LabeledAxisAccessor declare_subaxis(LabeledAxis & axis, const LabeledAxisAccessor & subaxis) const
  {
    axis.add<LabeledAxis>(subaxis);
    return subaxis;
  }

private:
  // Create a variable view (doesn't setup the view)
  template <typename T>
  Variable<T> * create_variable_view(Storage<LabeledAxisAccessor, VariableBase> & views,
                                     const LabeledAxisAccessor & name,
                                     TorchSize sz = -1)
  {
    if constexpr (std::is_same_v<T, BatchTensor>)
      neml_assert(sz > 0, "Allocating a BatchTensor requires a known storage size.");

    // Make sure we don't duplicate variable allocation
    VariableBase * var_base_ptr = views.query_value(name);
    neml_assert(!var_base_ptr,
                "Trying to allocate variable ",
                name,
                ", but a variable with the same name already exists.");

    // Allocate
    if constexpr (std::is_same_v<T, BatchTensor>)
    {
      auto var = std::make_unique<Variable<BatchTensor>>(name, sz);
      var_base_ptr = views.set_pointer(name, std::move(var));
    }
    else
    {
      auto var = std::make_unique<Variable<T>>(name);
      var_base_ptr = views.set_pointer(name, std::move(var));
    }

    // Cast it to the concrete type
    auto var_ptr = dynamic_cast<Variable<T> *>(var_base_ptr);
    neml_assert(
        var_ptr, "Internal error: Failed to cast variable ", name, " to its concrete type.");

    return var_ptr;
  }

  NEML2Object * _object;

  /**
   * @brief Parsed input file options. These options could be convenient when we look up a
   * cross-referenced tensor value by its name.
   *
   */
  // const OptionSet & _options;

  /// All the declared axes
  Storage<std::string, LabeledAxis> _axes;

  /// Input variable views
  Storage<LabeledAxisAccessor, VariableBase> _input_views;

  /// Output variable views
  Storage<LabeledAxisAccessor, VariableBase> _output_views;

  /// The input axis
  LabeledAxis & _input_axis;

  /// The output axis
  LabeledAxis & _output_axis;

  /// The storage for input variable values
  LabeledVector _in;

  /// The storage for output variable values
  LabeledVector _out;

  /// The storage for output variable 1st derivatives w.r.t. input variables
  LabeledMatrix _dout_din;

  /// The storage for output variable 2nd derivatives w.r.t. input variables
  LabeledTensor3D _d2out_din2;
};
} // namespace neml2