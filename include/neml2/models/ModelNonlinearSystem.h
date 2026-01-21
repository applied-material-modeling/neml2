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

#include "neml2/equation_systems/NonlinearSystem.h"

namespace neml2
{
class Model;

/// A monolith nonlinear system defined by a Model
class ModelNonlinearSystem : public NonlinearSystem
{
public:
  ModelNonlinearSystem(Model * model, bool assembly_guard = true);

  void set_u(const HVector &) override;
  void set_un(const HVector &) override;
  void set_g(const HVector &) override;
  void set_gn(const HVector &) override;

  HVector u() const override;
  HVector un() const override;
  HVector g() const override;
  HVector gn() const override;

  const Model & model() const { return *_model; }
  Model & model() { return *_model; }

  /// Whether this nonlinear system uses the AssemblyingNonlinearSystem guard when evaluating the model
  bool assembly_guard() const { return _assembly_guard; }

protected:
  void assemble(HMatrix *, HVector *) override;

  Model * _model;

  /// Whether to use the AssemblyingNonlinearSystem guard when evaluating the model
  bool _assembly_guard = true;
};

} // namespace neml2
