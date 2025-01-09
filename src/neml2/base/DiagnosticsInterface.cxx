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
#include "neml2/base/NEML2Object.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
Diagnosing::Diagnosing(bool ongoing)
  : prev_state(current_diagnostic_state())
{
  current_diagnostic_state().ongoing = ongoing;
}

Diagnosing::~Diagnosing() { current_diagnostic_state() = prev_state; }

DiagnosticState &
current_diagnostic_state()
{
  static DiagnosticState _diagnostic_state;
  return _diagnostic_state;
}

std::vector<Diagnosis> &
current_diagnoses()
{
  static std::vector<Diagnosis> _diagnoses;
  return _diagnoses;
}

std::vector<Diagnosis>
diagnose(const DiagnosticsInterface & patient)
{
  auto & state = current_diagnostic_state();
  state.patient_name = patient.object().name();
  state.patient_type = patient.object().type();

  // Run diagnostics
  {
    Diagnosing guard;
    patient.diagnose();
  }

  // Get the new diagnoses
  auto new_diagnoses = current_diagnoses();

  // Reset to a clean state before returning
  if (!state.ongoing)
  {
    current_diagnoses().clear();
    state.reset();
  }

  return new_diagnoses;
}

DiagnosticsInterface::DiagnosticsInterface(NEML2Object * object)
  : _object(object)
{
}
} // namespace neml2
