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
#include "neml2/solvers/NonlinearSolver.h"

namespace neml2
{
class ImplicitUpdate : public Model
{
public:
  static OptionSet expected_options();

  ImplicitUpdate(const OptionSet & options);

  void diagnose() const override;

  void link_output_variables() override;

  std::size_t last_iterations() const { return _last_iterations; }

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// The implicit model to be updated
  Model & _model;

  /// The nonlinear solver used to solve the nonlinear system
  std::shared_ptr<NonlinearSolver> _solver;

  /// Last solve result
  std::size_t _last_iterations = 0;
};
} // namespace neml2
