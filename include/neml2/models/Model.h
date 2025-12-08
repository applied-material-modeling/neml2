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

#include "neml2/models/DependencyDefinition.h"
#include "neml2/base/DiagnosticsInterface.h"

#include "neml2/models/Data.h"
#include "neml2/models/ParameterStore.h"
#include "neml2/models/VariableStore.h"
#include "neml2/models/NonlinearParameter.h"
#include "neml2/tensors/jit.h"

// These headers are not directly used by Model, but are included here so that derived classes do
// not have to include them separately. This is a convenience for the user, and is a reasonable
// choice since these headers are light and bring in little dependency.
#include "neml2/base/LabeledAxis.h"
#include "neml2/models/Variable.h"
#include "neml2/tensors/Derivative.h"

namespace neml2
{
class Model;

// Guard a region where implicit solve is being performed
struct AssemblyingNonlinearSystem
{
  AssemblyingNonlinearSystem(Model * const, bool assembling = true);

  AssemblyingNonlinearSystem(const AssemblyingNonlinearSystem &) = delete;
  AssemblyingNonlinearSystem(AssemblyingNonlinearSystem &&) = delete;
  AssemblyingNonlinearSystem & operator=(const AssemblyingNonlinearSystem &) = delete;
  AssemblyingNonlinearSystem & operator=(AssemblyingNonlinearSystem &&) = delete;
  ~AssemblyingNonlinearSystem();

  Model * const model;
  const bool prev_bool;
};

/**
 * @brief The base class for all constitutive models.
 *
 * A model maps some input to output. The forward operator (and its derivative) is defined in the
 * method `Model::set_value`. All concrete models must provide the implementation of the forward
 * operator by overriding the `Model::set_value` method.
 *
 * \hideinheritancegraph
 */
class Model : public std::enable_shared_from_this<Model>,
              public Data,
              public ParameterStore,
              public VariableStore,
              public DependencyDefinition<VariableName>,
              public DiagnosticsInterface
{
public:
  /**
   * @brief Schema for the traced forward operators
   *
   * The schema is determined by the dynamic dimensions of all input variables and model parameters.
   */
  struct EvaluationSchema
  {
    std::vector<Size> dynamic_dims;
    std::vector<TensorShape> intmd_shapes;
    at::DispatchKey dispatch_key;
    bool operator==(const EvaluationSchema & other) const;
    bool operator!=(const EvaluationSchema & other) const;
    bool operator<(const EvaluationSchema & other) const;
  };

  static OptionSet expected_options();

  /**
   * @brief Construct a new Model object
   *
   * @param options The options extracted from the input file
   */
  Model(const OptionSet & options);

  void setup() override;

  void diagnose() const override;

  /// Whether this model defines output values
  virtual bool defines_values() const { return _defines_value; }

  /// Whether this model defines first derivatives
  virtual bool defines_derivatives() const { return _defines_dvalue; }

  /// Whether this model defines second derivatives
  virtual bool defines_second_derivatives() const { return _defines_d2value; }

  ///@{
  /// Set or get whether we are currently assembling the nonlinear system
  void currently_assembling_nonlinear_system(bool);
  bool currently_assembling_nonlinear_system() const;
  ///@}

  /// Whether JIT is enabled
  virtual bool is_jit_enabled() const { return _jit; }

  /// Send model to a different device or dtype
  virtual void to(const TensorOptions & options);

  /// The models that may be used during the evaluation of this model
  const std::vector<std::shared_ptr<Model>> & registered_models() const
  {
    return _registered_models;
  }
  /// Get a registered model by its name
  std::shared_ptr<Model> registered_model(const std::string & name) const;

  /// Register a nonlinear parameter
  void register_nonlinear_parameter(const std::string & pname, const NonlinearParameter & param);

  /// Whether this parameter store has any nonlinear parameter
  bool has_nl_param(bool recursive = false) const;

  /**
   * @brief Query the existence of a nonlinear parameter
   *
   * @return const VariableBase* Pointer to the VariableBase if the parameter associated with the
   * given parameter name is nonlinear. Returns nullptr otherwise.
   */
  const VariableBase * nl_param(const std::string &) const;

  /// Get all nonlinear parameters
  virtual std::map<std::string, NonlinearParameter>
  named_nonlinear_parameters(bool recursive = false) const;

  /// The variables that this model depends on
  std::set<VariableName> consumed_items() const override;
  /// The variables that this model defines as part of its output
  std::set<VariableName> provided_items() const override;

  /// Request to use AD to compute the first derivative of a variable
  void request_AD(VariableBase & y, const VariableBase & u);

  /// Request to use AD to compute the second derivative of a variable
  void request_AD(VariableBase & y, const VariableBase & u1, const VariableBase & u2);

  void clear_input() override;
  void clear_output() override;
  void zero_undefined_input() override;

  /**
   * @brief Forward operator with jit
   *
   * If _jit is false, this falls back to the non-jit version.
   *
   * If _jit is true, it will use the corresponding traced graph as the forward operator,
   * and if the corresponding traced graph does not exists, it will create one.
   */
  void forward_maybe_jit(bool out, bool dout, bool d2out);

  /// Look up the name of a variable in the traced graph
  std::string variable_name_lookup(const ATensor & var) const;

  /// Convenient shortcut to construct and return the model value
  virtual ValueMap value(const ValueMap & in);

  /// Convenient shortcut to construct and return the model value and its derivative
  virtual std::tuple<ValueMap, DerivMap> value_and_dvalue(const ValueMap & in);

  /// Convenient shortcut to construct and return the derivative
  virtual DerivMap dvalue(const ValueMap & in);

  /// Convenient shortcut to construct and return the model's value, first and second derivative
  virtual std::tuple<ValueMap, DerivMap, SecDerivMap>
  value_and_dvalue_and_d2value(const ValueMap & in);

  /// Convenient shortcut to construct and return the model's second derivative
  virtual SecDerivMap d2value(const ValueMap & in);

  /// Convenient shortcut to construct and return the model's first and second derivative
  virtual std::tuple<DerivMap, SecDerivMap> dvalue_and_d2value(const ValueMap & in);

  /// Declaration of nonlinear parameters may require manipulation of input
  friend class ParameterStore;

  /// ComposedModel's set_value need to call submodel's set_value
  friend class ComposedModel;

protected:
  void diagnostic_assert_state(const VariableBase & v) const;
  void diagnostic_assert_old_state(const VariableBase & v) const;
  void diagnostic_assert_force(const VariableBase & v) const;
  void diagnostic_assert_old_force(const VariableBase & v) const;
  void diagnostic_assert_residual(const VariableBase & v) const;
  void diagnostic_check_input_variable(const VariableBase & v) const;
  void diagnostic_check_output_variable(const VariableBase & v) const;

  virtual void link_input_variables();
  virtual void link_input_variables(Model * submodel);
  virtual void link_output_variables();
  virtual void link_output_variables(Model * submodel);

  /// Check the current default precision and warn if it's not double precision
  void check_precision() const;

  /**
   * Request the use of automatic differentiation to compute variable derivatives
   *
   * Model implementations which require automatic differentiation to compute variable derivatives
   * shall override this method and mark variable derivatives. Variable derivatives are marked as,
   * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~cpp
   * // To request first derivative of foo with respect to bar
   * request_AD(foo, bar);
   *
   * // To request second derivative of foo with respect to bar and baz
   * request_AD(foo, bar, baz);
   * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   */
  virtual void request_AD() {}

  /// Forward operator without jit
  void forward(bool out, bool dout, bool d2out);

  /// The map between input -> output, and optionally its derivatives
  virtual void set_value(bool out, bool dout_din, bool d2out_din2) = 0;

  /**
   * @brief Register a model that the current model may use during its evaluation.
   *
   * If \p merge_input is set to true, this model will also *consume* the consumed variables of \p
   * model, which will affect dependency resolution inside a ComposedModel.
   *
   * @param name The model to register
   * @param merge_input Whether to merge the input axis of the registered model into *this* model's
   * input axis. This will make sure that the input variables of the registered model are "ready" by
   * the time *this* model is evaluated.
   */
  template <typename T = Model, typename = typename std::enable_if_t<std::is_base_of_v<Model, T>>>
  T & register_model(const std::string & name, bool merge_input = true)
  {
    auto model_name =
        input_options().contains(name) ? input_options().get<std::string>(name) : name;
    if (model_name == this->name())
      throw SetupException("Model named '" + this->name() +
                           "' is trying to register itself as a sub-model. This is not allowed.");

    OptionSet extra_opts;
    extra_opts.set<NEML2Object *>("_host") = host();

    if (!host()->factory())
      throw SetupException("Internal error: Host object '" + host()->name() +
                           "' does not have a factory set.");
    auto model = host()->factory()->get_object<T>("Models", model_name, extra_opts);
    if (std::find(_registered_models.begin(), _registered_models.end(), model) !=
        _registered_models.end())
      throw SetupException("Model named '" + model_name + "' has already been registered.");

    if (merge_input)
      for (auto && [name, var] : model->input_variables())
        clone_input_variable(*var);

    _registered_models.push_back(model);
    return *model;
  }

  void assign_input_stack(jit::Stack & stack);

  jit::Stack collect_input_stack() const;

  /// Models *this* model may use during its evaluation
  std::vector<std::shared_ptr<Model>> _registered_models;

private:
  template <typename T>
  void forward_helper(T && in, bool out, bool dout, bool d2out)
  {
    check_precision();
    assign_input(std::forward<T>(in));
    zero_undefined_input();
    forward_maybe_jit(out, dout, d2out);
  }

  /// Turn on AD for variable derivatives requested in neml2::Model::request_AD
  void enable_AD();

  /// Extract the AD derivatives of the output variables
  void extract_AD_derivatives(bool dout, bool d2out);

  /// Get the traced function for the forward operator
  std::size_t forward_operator_index(bool out, bool dout, bool d2out) const;

  /// @brief Compute the current schema
  EvaluationSchema calculate_eval_schema() const;

  ///@{
  /// Whether this model defines the value, first derivative, and second derivative
  bool _defines_value;
  bool _defines_dvalue;
  bool _defines_d2value;
  ///@}

  /// Parameters whose values are provided by another model
  std::map<std::string, NonlinearParameter> _nl_params;

  ///@{
  /// The variables that are requested to be differentiated
  std::map<VariableBase *, std::set<const VariableBase *>> _ad_derivs;
  std::map<VariableBase *, std::map<const VariableBase *, std::set<const VariableBase *>>>
      _ad_secderivs;
  std::set<VariableBase *> _ad_args;
  ///@}

  /// Whether to use JIT
  const bool _jit;

  /// Whether to use production mode
  const bool _production;

  /**
   * @brief Cached function graphs and their schema for forward operators
   *
   * The index is the binary encoding of the tuple (out, dout, d2out)
   *
   * See the table below
   * Decimal index, Binary index, Value, Derivative, 2nd derivative
   * 0, 000, no, no, no  <-- We don't provide this API
   * 1, 001, no, no, yes
   * 2, 010, no, yes, no
   * 3, 011, no, yes, yes
   * 4, 100, yes, no, no
   * 5, 101, yes, no, yes  <-- We don't provide this API
   * 6, 110, yes, yes, no
   * 7, 111, yes, yes, yes
   */
  std::array<std::map<EvaluationSchema, std::unique_ptr<jit::GraphFunction>>, 8> _traced_functions;

  /// Similar to _trace_functions, but for the forward operator of the nonlinear system
  std::array<std::map<EvaluationSchema, std::unique_ptr<jit::GraphFunction>>, 8>
      _traced_functions_nl_sys;

  /// Whether we are currently assembling a nonlinear system
  bool _currently_assembling_nonlinear_system = false;
};

std::ostream & operator<<(std::ostream & os, const Model & model);
} // namespace neml2
