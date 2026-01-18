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

#include "neml2/solvers/NonlinearSystem.h"

namespace neml2
{
OptionSet
NonlinearSystem::expected_options()
{
  OptionSet options;

  return options;
}

NonlinearSystem::NonlinearSystem(const OptionSet & /*options*/) {}

void
NonlinearSystem::set_umap(const std::vector<LabeledAxisAccessor> & unknowns,
                          const std::vector<TensorShapeRef> & unknown_shapes)
{
  _unknowns = unknowns;
  _unknown_shapes = {unknown_shapes.begin(), unknown_shapes.end()};
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::umap() const
{
  return _unknowns;
}

const std::vector<TensorShape> &
NonlinearSystem::ulayout() const
{
  return _unknown_shapes;
}

void
NonlinearSystem::set_unmap(const std::vector<LabeledAxisAccessor> & old_solutions,
                           const std::vector<TensorShapeRef> & old_solution_shapes)
{
  _old_solutions = old_solutions;
  _old_solution_shapes = {old_solution_shapes.begin(), old_solution_shapes.end()};
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::unmap() const
{
  return _old_solutions;
}

const std::vector<TensorShape> &
NonlinearSystem::unlayout() const
{
  return _old_solution_shapes;
}

void
NonlinearSystem::set_gmap(const std::vector<LabeledAxisAccessor> & given,
                          const std::vector<TensorShapeRef> & given_shapes)
{
  _given = given;
  _given_shapes = {given_shapes.begin(), given_shapes.end()};
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::gmap() const
{
  return _given;
}

const std::vector<TensorShape> &
NonlinearSystem::glayout() const
{
  return _given_shapes;
}

void
NonlinearSystem::set_gnmap(const std::vector<LabeledAxisAccessor> & old_given,
                           const std::vector<TensorShapeRef> & old_given_shapes)
{
  _old_given = old_given;
  _old_given_shapes = {old_given_shapes.begin(), old_given_shapes.end()};
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::gnmap() const
{
  return _old_given;
}

const std::vector<TensorShape> &
NonlinearSystem::gnlayout() const
{
  return _old_given_shapes;
}

void
NonlinearSystem::set_rmap(const std::vector<LabeledAxisAccessor> & residuals,
                          const std::vector<TensorShapeRef> & residual_shapes)
{
  _residuals = residuals;
  _residual_shapes = {residual_shapes.begin(), residual_shapes.end()};
}

const std::vector<LabeledAxisAccessor> &
NonlinearSystem::rmap() const
{
  return _residuals;
}

const std::vector<TensorShape> &
NonlinearSystem::rlayout() const
{
  return _residual_shapes;
}

es::Vector
NonlinearSystem::create_uvec() const
{
  return es::Vector(_unknown_shapes);
}

es::Vector
NonlinearSystem::create_unvec() const
{
  return es::Vector(_old_solution_shapes);
}

es::Vector
NonlinearSystem::create_gvec() const
{
  return es::Vector(_given_shapes);
}

es::Vector
NonlinearSystem::create_gnvec() const
{
  return es::Vector(_old_given_shapes);
}

es::Vector
NonlinearSystem::create_rvec() const
{
  return es::Vector(_residual_shapes);
}

es::Vector
NonlinearSystem::residual()
{
  es::Vector r;
  assemble(&r, nullptr);
  return r;
}

es::Vector
NonlinearSystem::residual(const es::Vector & x)
{
  set_solution(x);
  return residual();
}

es::Matrix
NonlinearSystem::Jacobian()
{
  es::Matrix J;
  assemble(nullptr, &J);
  return J;
}

es::Matrix
NonlinearSystem::Jacobian(const es::Vector & x)
{
  set_solution(x);
  return Jacobian();
}

std::tuple<es::Vector, es::Matrix>
NonlinearSystem::residual_and_Jacobian()
{
  es::Vector r;
  es::Matrix J;
  assemble(&r, &J);
  return {r, J};
}

std::tuple<es::Vector, es::Matrix>
NonlinearSystem::residual_and_Jacobian(const es::Vector & x)
{
  set_solution(x);
  return residual_and_Jacobian();
}
} // namespace neml2
