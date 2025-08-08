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

#include "neml2/misc/errors.h"
#include "neml2/misc/string_utils.h"

namespace neml2
{
// Forward decl
class DiagnosticsInterface;
class VariableBase;
class NEML2Object;

struct DiagnosticState
{
  bool ongoing = false;
  std::string patient_name = "";
  std::string patient_type = "";

  void reset()
  {
    ongoing = false;
    patient_name.clear();
    patient_type.clear();
  }
};

// Guard a region when diagnostics are being performed
struct Diagnosing
{
  Diagnosing(bool ongoing = true);

  Diagnosing(const Diagnosing &) = delete;
  Diagnosing(Diagnosing &&) = delete;
  Diagnosing & operator=(const Diagnosing &) = delete;
  Diagnosing & operator=(Diagnosing &&) = delete;
  ~Diagnosing();

  const DiagnosticState prev_state;
};

/// Get the current diagnostic state
DiagnosticState & current_diagnostic_state();

/// Get the current diagnoses
std::vector<Diagnosis> & current_diagnoses();

/// A helper function to diagnose common setup errors
std::vector<Diagnosis> diagnose(const DiagnosticsInterface &);

/// A helper function to diagnose common setup errors and throw an exception if any errors are found
void diagnose_and_throw(const DiagnosticsInterface &);

/// Helper assertion function for diagnostics
template <typename... Args>
void diagnostic_assert(bool, Args &&...);

/// Interface for object making diagnostics about common setup errors
class DiagnosticsInterface
{
public:
  DiagnosticsInterface() = delete;
  DiagnosticsInterface(NEML2Object * object);

  DiagnosticsInterface(DiagnosticsInterface &&) = delete;
  DiagnosticsInterface(const DiagnosticsInterface &) = delete;
  DiagnosticsInterface & operator=(const DiagnosticsInterface &) = delete;
  DiagnosticsInterface & operator=(DiagnosticsInterface &&) = delete;
  virtual ~DiagnosticsInterface() = default;

  /**
   * @brief Check for common problems
   *
   * This method serves as the entry point for diagnosing common problems in object setup. The idea
   * behind this method is that while some errors could be detected at construction time, i.e., when
   * the object's constructor is called, it doesn't hinder other objects' creation. We therefore
   * would like to defer the detection of errors until after all objects have been created, collect
   * all errors at once, and present the user with a complete understanding of all errors
   * encountered.
   *
   * Note, however, if an error could interfere with other objects' creation, it should be raised
   * right away inside the constructor, instead of inside this method.
   */
  virtual void diagnose() const = 0;

  /// Get the object
  const NEML2Object & object() const { return *_object; }

private:
  NEML2Object * _object;
};
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
template <typename... Args>
void
diagnostic_assert(bool assertion, Args &&... args)
{
  if (assertion)
    return;

  auto & state = current_diagnostic_state();

  if (!state.ongoing)
    throw NEMLException("Diagnostics are not currently being run. diagnostic_assert should only be "
                        "called inside a DiagnosticsInterface::diagnose method.");

  std::ostringstream oss;
  utils::stream_all(oss,
                    "In object '",
                    state.patient_name,
                    "' of type ",
                    state.patient_type,
                    ": ",
                    std::forward<Args>(args)...);

  current_diagnoses().emplace_back(oss.str());
}
} // namespace neml2
