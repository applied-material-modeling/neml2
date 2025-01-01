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

#include <torch/csrc/jit/api/function_impl.h>

#include "neml2/base/DependencyDefinition.h"
#include "neml2/base/DiagnosticsInterface.h"

#include "neml2/models/Data.h"
#include "neml2/models/ParameterStore.h"
#include "neml2/models/VariableStore.h"
#include "neml2/solvers/NonlinearSystem.h"

namespace neml2
{
/**
 * @brief The base class for all constitutive models.
 *
 * A model maps some input to output. The forward operator (and its derivative) is defined in the
 * method `Model::set_value`. All concrete models must provide the implementation of the forward
 * operator by overriding the `Model::set_value` method.
 */
class Model : public std::enable_shared_from_this<Model>,
              public Data,
              public ParameterStore,
              public VariableStore,
              public NonlinearSystem,
              public DependencyDefinition<VariableName>,
              public DiagnosticsInterface
{
public:
  /**
   * @brief Schema for the traced forward operators
   *
   * The schema is determined by the batch dimensions of all input variables and model parameters.
   */
  struct TraceSchema
  {
    std::vector<Size> batch_dims;
    bool operator==(const TraceSchema & other) const { return batch_dims == other.batch_dims; }
  };

  static OptionSet expected_options();

  /**
   * @brief Construct a new Model object
   *
   * @param options The options extracted from the input file
   */
  Model(const OptionSet & options);

  /// Send model to a different device or dtype
  virtual void to(const torch::TensorOptions & options);

  void diagnose(std::vector<Diagnosis> &) const override;

  /// Whether this model defines one or more nonlinear equations to be solved
  virtual bool is_nonlinear_system() const { return _nonlinear_system; }

  /// Whether JIT is enabled
  virtual bool is_jit_enabled() const { return _jit; }

  /// The models that may be used during the evaluation of this model
  const std::vector<Model *> & registered_models() const { return _registered_models; }
  /// Get a registered model by its name
  Model * registered_model(const std::string & name) const;

  /// The variables that this model depends on
  std::set<VariableName> consumed_items() const override;
  /// The variables that this model defines as part of its output
  std::set<VariableName> provided_items() const override;

  /// Request to use AD to compute the first derivative of a variable
  void request_AD(VariableBase & y, const VariableBase & u);

  /// Request to use AD to compute the second derivative of a variable
  void request_AD(VariableBase & y, const VariableBase & u1, const VariableBase & u2);

  /// Forward operator without jit
  void forward(bool out, bool dout, bool d2out);

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
  std::string variable_name_lookup(const torch::Tensor & var);

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

  /// Get the last traced function for the forward operator
  const torch::jit::GraphFunction * traced_function(bool out, bool dout, bool d2out) const;

  /// Get the last traced function for the forward operator of the nonlinear system
  const torch::jit::GraphFunction * traced_function_nl_sys(bool out, bool dout, bool d2out) const;

  /// Declaration of nonlinear parameters may require manipulation of input
  friend class ParameterStore;

  /// ComposedModel's set_value need to call submodel's set_value
  friend class ComposedModel;

protected:
  void setup() override;
  virtual void link_input_variables();
  virtual void link_input_variables(Model * submodel);
  virtual void link_output_variables();
  virtual void link_output_variables(Model * submodel);

  void clear_input() override;
  void clear_output() override;
  void zero_input() override;
  void zero_output() override;

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

  /// Additional diagnostics for a nonlinear system
  void diagnose_nl_sys(std::vector<Diagnosis> & diagnoses) const;

  /// The map between input -> output, and optionally its derivatives
  virtual void set_value(bool out, bool dout_din, bool d2out_din2) = 0;

  /**
   * @brief Register a model that the current model may use during its evaluation.
   *
   * If \p merge_input is set to true, this model will also *consume* the consumed variables of \p
   * model, which will affect dependency resolution inside a ComposedModel.
   *
   * @param name The model to register
   * @param nonlinear Set to true if the registered model defines a nonlinear system to be solved
   * @param merge_input Whether to merge the input axis of the registered model into *this* model's
   * input axis. This will make sure that the input variables of the registered model are "ready" by
   * the time *this* model is evaluated.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Model, T>>>
  T & register_model(const std::string & name, bool nonlinear = false, bool merge_input = true)
  {
    neml_assert(name != this->name(),
                "Model named '",
                this->name(),
                "' is trying to register itself as a sub-model. This is not allowed.");

    OptionSet extra_opts;
    extra_opts.set<NEML2Object *>("_host") = host();
    extra_opts.set<bool>("_nonlinear_system") = nonlinear;

    auto model = Factory::get_object_ptr<Model>("Models", name, extra_opts);

    if (merge_input)
      for (auto && [name, var] : model->input_variables())
        clone_input_variable(var);

    _registered_models.push_back(model.get());
    return *(std::dynamic_pointer_cast<T>(model));
  }

  void assign_input_stack(torch::jit::Stack & stack);

  torch::jit::Stack collect_input_stack() const;

  void set_guess(const Sol<false> &) override;

  void assemble(Res<false> *, Jac<false> *) override;

  /// Models *this* model may use during its evaluation
  std::vector<Model *> _registered_models;

private:
  /// Given the requested AD derivatives, should the forward operator
  /// neml2::Model::set_value compute the output variable?
  bool AD_need_value(bool dout, bool d2out) const;

  /// Turn on AD for variable derivatives requested in neml2::Model::request_AD
  void enable_AD();

  /// Extract the AD derivatives of the output variables
  void extract_AD_derivatives(bool dout, bool d2out);

  /// Get the traced function for the forward operator
  std::size_t forward_operator_index(bool out, bool dout, bool d2out) const;

  /// Compute the trace schema
  TraceSchema compute_trace_schema() const;

  /// Whether this is a nonlinear system
  bool _nonlinear_system;

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
  std::array<std::pair<TraceSchema, std::unique_ptr<torch::jit::GraphFunction>>, 8>
      _traced_functions;

  /// Similar to _trace_functions, but for the forward operator of the nonlinear system
  std::array<std::pair<TraceSchema, std::unique_ptr<torch::jit::GraphFunction>>, 8>
      _traced_functions_nl_sys;
};

std::ostream & operator<<(std::ostream & os, const Model & model);
} // namespace neml2
