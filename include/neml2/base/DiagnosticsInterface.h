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

#include "neml2/base/NEML2Object.h"

namespace neml2
{
// Forward decl
class DiagnosticsInterface;
class VariableBase;

/// Exception type reserved for diagnostics, so as to not conceptually clash with other exceptions
class Diagnosis : public NEMLException
{
public:
  using NEMLException::NEMLException;
};

/// Raise diagnostics as exception, if any
void diagnose(const DiagnosticsInterface &);

/// Interface for object making diagnostics about common setup errors
class DiagnosticsInterface
{
public:
  DiagnosticsInterface(NEML2Object * object);

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
   *
   * @param diagnoses A vector of exceptions of type Diagnosis for each of the detected problem.
   */
  virtual void diagnose(std::vector<Diagnosis> & diagnoses) const = 0;

  template <typename... Args>
  void diagnostic_assert(std::vector<Diagnosis> & diagnoses, bool assertion, Args &&... args) const;

  void diagnostic_assert_state(std::vector<Diagnosis> & diagnoses, const VariableBase & v) const;
  void diagnostic_assert_old_state(std::vector<Diagnosis> & diagnoses,
                                   const VariableBase & v) const;
  void diagnostic_assert_force(std::vector<Diagnosis> & diagnoses, const VariableBase & v) const;
  void diagnostic_assert_old_force(std::vector<Diagnosis> & diagnoses,
                                   const VariableBase & v) const;
  void diagnostic_assert_residual(std::vector<Diagnosis> & diagnoses, const VariableBase & v) const;
  void diagnostic_check_variable(std::vector<Diagnosis> & diagnoses, const VariableBase & v) const;

private:
  NEML2Object * _object;
};

template <typename... Args>
void
DiagnosticsInterface::diagnostic_assert(std::vector<Diagnosis> & diagnoses,
                                        bool assertion,
                                        Args &&... args) const
{
  if (assertion)
    return;

  std::ostringstream oss;
  internal::stream_all(oss,
                       "In object '",
                       _object->name(),
                       "' of type ",
                       _object->type(),
                       ": ",
                       std::forward<Args>(args)...);
  return diagnoses.push_back(Diagnosis(oss.str().data()));
}
}
