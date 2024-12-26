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

#include "neml2/models/Model.h"
#include "neml2/base/DependencyResolver.h"

#include <future>

namespace neml2
{
class ComposedModel : public Model
{
public:
  static OptionSet expected_options();

  ComposedModel(const OptionSet & options);

  std::map<std::string, const VariableBase *>
  named_nonlinear_parameters(bool recursive = false) const override;

  std::map<std::string, Model *>
  named_nonlinear_parameter_models(bool recursive = false) const override;

  bool is_jit_enabled() const override { return _jit; }

protected:
  void link_input_variables(Model * submodel) override;
  void link_output_variables(Model * submodel) override;
  void set_value(bool, bool, bool) override;

private:
  /// Additional outbound items in the dependency graph
  const std::vector<VariableName> _additional_outputs;

  /// Whether to automatically add nonlinear parameters
  const bool _auto_nl_param;

  /// Helper to resolve model dependency
  DependencyResolver<Model, VariableName> _dependency;

  /// Whether JIT is enabled
  bool _jit;
};
} // namespace neml2
