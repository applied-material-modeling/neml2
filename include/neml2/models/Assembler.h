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

#include "neml2/models/LabeledAxis.h"
#include "neml2/tensors/Tensor.h"

namespace neml2
{
/**
 * @brief Helper to assemble a vector of tensors into a single tensor and also to disassemble
 * a tensor into a map of tensors.
 *
 */
class VectorAssembler
{
public:
  VectorAssembler(const LabeledAxis & axis)
    : _axis(axis)
  {
  }

  /// Assemble a vector of vectors (by variables)
  Tensor assemble(const std::map<VariableName, Tensor> &) const;

  /// Split the vector (by variables)
  std::map<VariableName, Tensor> disassemble(const Tensor &) const;

  /// Split the vector (by subaxes)
  std::map<SubaxisName, Tensor> split(const Tensor &) const;

private:
  const LabeledAxis & _axis;
};

/**
 * @brief Helper to assemble a matrix of tensors into a single tensor and also to disassemble
 * a tensor into a map of map of tensors.
 *
 */
class MatrixAssembler
{
public:
  MatrixAssembler(const LabeledAxis & yaxis, const LabeledAxis & xaxis)
    : _yaxis(yaxis),
      _xaxis(xaxis)
  {
  }

  /// Assemble a matrix of matrices (by variables)
  Tensor assemble(const std::map<VariableName, std::map<VariableName, Tensor>> &) const;

  /// Split the matrix (by variables)
  std::map<VariableName, std::map<VariableName, Tensor>> disassemble(const Tensor &) const;

  /// Split the matrix (by subaxes)
  std::map<SubaxisName, std::map<SubaxisName, Tensor>> split(const Tensor &) const;

private:
  const LabeledAxis & _yaxis;
  const LabeledAxis & _xaxis;
};
} // namespace neml2
