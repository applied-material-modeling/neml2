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

#include "neml2/drivers/Driver.h"
#include "neml2/jit/types.h"

namespace neml2
{
class TransientDriver;

class TransientRegression : public Driver
{
public:
  static OptionSet expected_options();

  TransientRegression(const OptionSet & options);

  void diagnose() const override;

  bool run() override;

private:
  /// The driver that will run the NEML2 model
  const std::shared_ptr<TransientDriver> _driver;

  /// The reference file to be diff'ed against
  std::filesystem::path _reference;

  double _rtol;
  double _atol;
};

std::string diff(const jit::named_buffer_list & res,
                 const jit::named_buffer_list & ref,
                 double rtol = 1e-5,
                 double atol = 1e-8);
} // namespace neml2
