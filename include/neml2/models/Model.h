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
 * method \p set_value. All concrete models must provide the implementation of the forward operator
 * by overriding the \p set_value method.
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
  static OptionSet expected_options();

  /**
   * @brief Construct a new Model object
   *
   * @param options The options extracted from the input file
   */
  Model(const OptionSet & options);

  /// Send model to a different device or dtype
  virtual void to(const torch::TensorOptions & options);

  virtual void diagnose(std::vector<Diagnosis> &) const override;

  /// Whether this model defines one or more nonlinear equations to be solved
  virtual bool is_nonlinear_system() const { return _nonlinear_system; }

  /// The models that may be used during the evaluation of this model
  const std::vector<Model *> & registered_models() const { return _registered_models; }
  /// Get a registered model by its name
  Model * registered_model(const std::string & name) const;

  /// The variables that this model depends on
  virtual std::set<VariableName> consumed_items() const override;
  /// The variables that this model defines as part of its output
  virtual std::set<VariableName> provided_items() const override;

  void clear_input() override;
  void clear_output() override;
  void zero_input() override;
  void zero_output() override;

  /// Request to use AD to compute the first derivative of a variable
  void request_AD(VariableBase & y, const VariableBase & u);

  /// Request to use AD to compute the second derivative of a variable
  void request_AD(VariableBase & y, const VariableBase & u1, const VariableBase & u2);

  /// Evalute the model
  virtual void value();
  /// Evalute the model and compute its derivative
  virtual void value_and_dvalue();
  /// Evalute the derivative
  virtual void dvalue();
  /// Evalute the model and compute its first and second derivatives
  virtual void value_and_dvalue_and_d2value();
  /// Evalute the second derivatives
  virtual void d2value();
  /// Evalute the first and second derivatives
  virtual void dvalue_and_d2value();

  /// Convenient shortcut to construct and return the model value
  virtual std::map<VariableName, Tensor> value(const std::map<VariableName, Tensor> & in);

  /// Convenient shortcut to construct and return the model value and its derivative
  virtual std::tuple<std::map<VariableName, Tensor>,
                     std::map<VariableName, std::map<VariableName, Tensor>>>
  value_and_dvalue(const std::map<VariableName, Tensor> & in);

  /// Convenient shortcut to construct and return the derivative
  virtual std::map<VariableName, std::map<VariableName, Tensor>>
  dvalue(const std::map<VariableName, Tensor> & in);

  /// Convenient shortcut to construct and return the model's value, first and second derivative
  virtual std::tuple<std::map<VariableName, Tensor>,
                     std::map<VariableName, std::map<VariableName, Tensor>>,
                     std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>>>
  value_and_dvalue_and_d2value(const std::map<VariableName, Tensor> & in);

  /// Convenient shortcut to construct and return the model's second derivative
  virtual std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>>
  d2value(const std::map<VariableName, Tensor> & in);

  /// Convenient shortcut to construct and return the model's first and second derivative
  virtual std::tuple<std::map<VariableName, std::map<VariableName, Tensor>>,
                     std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>>>
  dvalue_and_d2value(const std::map<VariableName, Tensor> & in);

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

  void set_guess(const SOL<false> &) override;

  void assemble(RES<false> *, JAC<false> *) override;

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

  /// Whether this is a nonlinear system
  bool _nonlinear_system;

  /// The variables that are requested to be differentiated
  std::map<VariableBase *, std::set<const VariableBase *>> _ad_derivs;
  std::map<VariableBase *, std::map<const VariableBase *, std::set<const VariableBase *>>>
      _ad_secderivs;
  std::set<VariableBase *> _ad_args;
};

std::ostream & operator<<(std::ostream & os, const Model & model);
} // namespace neml2
