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

#include <filesystem>

#include "neml2/base/Registry.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/base/Factory.h"
#include "neml2/base/DiagnosticsInterface.h"
#include "neml2/models/Model.h"

#ifdef NEML2_HAS_DISPATCHER
#include "neml2/dispatchers/WorkScheduler.h"
#endif

// The following are not directly used by Solver itself.
// We put them here so that derived classes can add expected options of these types.
#include "neml2/base/TensorName.h"
#include "neml2/base/EnumSelection.h"

namespace neml2
{
class Driver;

/**
 * @brief A convenient function to manufacture a neml2::Driver
 *
 * The input file must have already been parsed and loaded.
 *
 * @param dname Name of the driver
 */
Driver & get_driver(const std::string & dname);

/**
 * @brief The Driver drives the execution of a NEML2 Model.
 *
 */
class Driver : public NEML2Object, public DiagnosticsInterface
{
public:
  static OptionSet expected_options();

  /**
   * @brief Construct a new Driver object
   *
   * @param options The options extracted from the input file
   */
  Driver(const OptionSet & options);

  void diagnose() const override {}

  /// Let the driver run, return \p true upon successful completion, and return \p false otherwise.
  virtual bool run() = 0;

protected:
  /// Whether to print out additional (debugging) information during the execution.
  bool _verbose;

#ifdef NEML2_HAS_DISPATCHER
  std::shared_ptr<WorkScheduler> _scheduler;
  const bool _async_dispatch;
#endif
};
} // namespace neml2
