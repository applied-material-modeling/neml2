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

#include "neml2/base/DiagnosticsInterface.h"
#include "neml2/models/Variable.h"

namespace neml2
{
void
diagnose(const DiagnosticsInterface & patient)
{
  std::vector<Diagnosis> diagnoses;
  patient.diagnose(diagnoses);

  if (!diagnoses.empty())
  {
    std::stringstream message;
    for (auto & diagnosis : diagnoses)
      message << diagnosis.what() << "\n\n";
    throw NEMLException(message.str());
  }
}

DiagnosticsInterface::DiagnosticsInterface(NEML2Object * object)
  : _object(object)
{
}

void
DiagnosticsInterface::diagnostic_assert_state(std::vector<Diagnosis> & diagnoses,
                                              const VariableBase & v) const
{
  diagnostic_assert(
      diagnoses, v.is_state(), "Variable ", v.name(), " must be on the state sub-axis.");
}

void
DiagnosticsInterface::diagnostic_assert_old_state(std::vector<Diagnosis> & diagnoses,
                                                  const VariableBase & v) const
{
  diagnostic_assert(
      diagnoses, v.is_old_state(), "Variable ", v.name(), " must be on the old_state sub-axis.");
}

void
DiagnosticsInterface::diagnostic_assert_force(std::vector<Diagnosis> & diagnoses,
                                              const VariableBase & v) const
{
  diagnostic_assert(
      diagnoses, v.is_force(), "Variable ", v.name(), " must be on the forces sub-axis.");
}

void
DiagnosticsInterface::diagnostic_assert_old_force(std::vector<Diagnosis> & diagnoses,
                                                  const VariableBase & v) const
{
  diagnostic_assert(
      diagnoses, v.is_old_force(), "Variable ", v.name(), " must be on the old_forces sub-axis.");
}

void
DiagnosticsInterface::diagnostic_assert_residual(std::vector<Diagnosis> & diagnoses,
                                                 const VariableBase & v) const
{
  diagnostic_assert(
      diagnoses, v.is_residual(), "Variable ", v.name(), " must be on the residual sub-axis.");
}

void
DiagnosticsInterface::diagnostic_check_input_variable(std::vector<Diagnosis> & diagnoses,
                                                      const VariableBase & v) const
{
  diagnostic_assert(diagnoses,
                    v.is_state() || v.is_old_state() || v.is_force() || v.is_old_force() ||
                        v.is_residual() || v.is_parameter(),
                    "Input variable ",
                    v.name(),
                    " must be on one of the following sub-axes: state, old_state, forces, "
                    "old_forces, residual, parameters.");
}

void
DiagnosticsInterface::diagnostic_check_output_variable(std::vector<Diagnosis> & diagnoses,
                                                       const VariableBase & v) const
{
  diagnostic_assert(
      diagnoses,
      v.is_state() || v.is_force() || v.is_residual() || v.is_parameter(),
      "Output variable ",
      v.name(),
      " must be on one of the following sub-axes: state, forces, residual, parameters.");
}
} // namespace neml2
