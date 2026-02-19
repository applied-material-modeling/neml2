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

#include <catch2/catch_test_macros.hpp>

#include "neml2/neml2.h"
#include "neml2/solvers/SchurComplement.h"
#include "neml2/models/ModelNonlinearSystem.h"

using namespace neml2;

TEST_CASE("SchurComplement", "[solvers]")
{
  SECTION("solver creation")
  {
    // Load the input file that defines the SchurComplement solver
    auto factory = load_input("solvers/schur_complement.i");

    // Check that we can get the solver
    auto solver = factory->get_solver<SchurComplement>("schur");
    REQUIRE(solver != nullptr);

    // Check that the nested nonlinear system can be created
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("eq_sys");
    REQUIRE(eq_sys != nullptr);

    // Check that the system has 2 groups
    REQUIRE(eq_sys->n_ugroup() == 2);
  }
}
