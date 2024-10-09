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

#include "neml2/solvers/NonlinearSystem.h"

namespace neml2
{
class TestNonlinearSystem : public NonlinearSystem
{
public:
  TestNonlinearSystem(const OptionSet & options);

  void set_guess(const SOL<false> & x) override;
  virtual Tensor exact_solution(const SOL<false> & x) const = 0;

protected:
  Tensor _x;
};

class PowerTestSystem : public TestNonlinearSystem
{
public:
  PowerTestSystem(const OptionSet & options);

  Tensor exact_solution(const SOL<false> & x) const override;

protected:
  void assemble(RES<false> *, JAC<false> *) override;
};

class RosenbrockTestSystem : public TestNonlinearSystem
{
public:
  RosenbrockTestSystem(const neml2::OptionSet & options);

  neml2::Tensor exact_solution(const SOL<false> & x) const override;

protected:
  void assemble(RES<false> *, JAC<false> *) override;
};
}
