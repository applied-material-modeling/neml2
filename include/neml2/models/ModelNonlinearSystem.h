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
#include "neml2/equation_systems/SparseTensorList.h"

namespace neml2
{
class Model;

/// A monolith nonlinear system defined by a Model
class ModelNonlinearSystem : public NonlinearSystem
{
public:
  static OptionSet expected_options();

  ModelNonlinearSystem(const OptionSet & options);

  const Model & model() const { return *_model; }
  Model & model() { return *_model; }

  void setup() override;

  void to(const TensorOptions &) override;

  void set_u(const SparseTensorList &) override;
  void set_g(const SparseTensorList &) override;

  SparseTensorList u() const override;
  SparseTensorList g() const override;
  SparseTensorList B() override;

protected:
  std::vector<LabeledAxisAccessor> setup_umap() override;
  std::vector<TensorShape> setup_intmd_ulayout() override;
  std::vector<TensorShape> setup_ulayout() override;

  std::vector<LabeledAxisAccessor> setup_bmap() override;
  std::vector<TensorShape> setup_intmd_blayout() override;
  std::vector<TensorShape> setup_blayout() override;

  std::vector<LabeledAxisAccessor> setup_gmap() override;
  std::vector<TensorShape> setup_intmd_glayout() override;
  std::vector<TensorShape> setup_glayout() override;

  void assemble(SparseTensorList * A, SparseTensorList * b) override;

private:
  std::shared_ptr<Model> _model;
};

} // namespace neml2
