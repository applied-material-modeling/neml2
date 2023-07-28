// Copyright 2023, UChicago Argonne, LLC
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

#include "neml2/drivers/Driver.h"
#include "neml2/drivers/ResultSeriesContainer.h"
#include <filesystem>

namespace neml2
{
class TransientDriver : public Driver
{
public:
  static ParameterSet expected_params();

  TransientDriver(const ParameterSet & params);

  bool run() override;

  virtual std::string save_as_path() const;

  virtual std::shared_ptr<ResultSeriesContainer> result() const { return _result; }

protected:
  virtual void check_integrity() const override;
  virtual bool solve() = 0;
  virtual void output() const;
  void output_pt(const std::filesystem::path & out) const;

  Model & _model;

  torch::Tensor _time;
  TorchSize _step_count;
  LabeledAxisAccessor _time_name;
  TorchSize _nsteps;
  TorchSize _nbatch;
  LabeledVector _in;
  LabeledVector _out;

  std::string _save_as;
  const bool _show_params;
  std::shared_ptr<ResultSeriesContainer> _result;
};
} // namespace neml2
