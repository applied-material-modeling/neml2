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

  void set_solution(const HVector & x) override;
  HVector get_solution() const override { return _x; }
  virtual HVector exact_solution(const HVector & x) const = 0;

protected:
  HVector _x;
};

class PowerTestSystem : public TestNonlinearSystem
{
public:
  PowerTestSystem(const OptionSet & options);

  HVector exact_solution(const HVector & x) const override;

protected:
  void assemble(HVector *, HMatrix *) override;
};

class RosenbrockTestSystem : public TestNonlinearSystem
{
public:
  RosenbrockTestSystem(const neml2::OptionSet & options);

  HVector exact_solution(const HVector & x) const override;

protected:
  void assemble(HVector *, HMatrix *) override;
};
}
