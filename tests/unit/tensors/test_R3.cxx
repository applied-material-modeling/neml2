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

#include "utils.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("R3", "[tensors]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape

  SECTION("class R3")
  {
    SECTION("levi_civita")
    {
      auto lc = R3::levi_civita(DTO);

      std::vector<TensorShape> p = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
      std::vector<TensorShape> n = {{2, 1, 0}, {1, 0, 2}, {0, 2, 1}};
      for (Size i = 0; i < 3; i++)
        for (Size j = 0; j < 3; j++)
          for (Size k = 0; k < 3; k++)
          {
            Real v = 0.0;
            if (std::find(p.begin(), p.end(), TensorShape{i, j, k}) != p.end())
              v = 1.0;
            else if (std::find(n.begin(), n.end(), TensorShape{i, j, k}) != n.end())
              v = -1.0;
            REQUIRE(at::allclose(lc(i, j, k), Scalar(v, DTO)));
          }
    }

    SECTION("operator()")
    {
      auto a = R3(at::rand(utils::add_shapes(B, 3, 3, 3), DTO));
      for (Size i = 0; i < 3; i++)
        for (Size j = 0; j < 3; j++)
          for (Size k = 0; k < 3; k++)
            REQUIRE(at::allclose(a(i, j, k), a.index({indexing::Ellipsis, i, j, k})));
    }

    SECTION("contract_k")
    {
      auto T = R3::create({{{0.2051969, 0.01953205, 0.46272625},
                            {0.76724114, 0.12504687, 0.66948082},
                            {0.13071273, 0.46393329, 0.36774737}},
                           {{0.37293691, 0.4930683, 0.14386892},
                            {0.75014405, 0.07264975, 0.36369278},
                            {0.22465346, 0.08398304, 0.75035687}},
                           {{0.34658198, 0.69665782, 0.38999866},
                            {0.18213956, 0.48116027, 0.9926462},
                            {0.4324153, 0.40872475, 0.93650606}}},
                          DTO);
      auto v = Vec::fill(0.81256887, 0.31300369, 0.01151858, DTO);
      auto res = R2::create({{0.17818016, 0.67028787, 0.25566185},
                             {0.45902629, 0.63647256, 0.21747645},
                             {0.50417043, 0.31003975, 0.49008678}},
                            DTO);
      REQUIRE(at::allclose(T.contract_k(v), res));
      REQUIRE(at::allclose(T.contract_k(v.batch_expand(B)), res.batch_expand(B)));
      REQUIRE(at::allclose(T.batch_expand(B).contract_k(v), res.batch_expand(B)));
      REQUIRE(at::allclose(T.batch_expand(B).contract_k(v.batch_expand(B)), res.batch_expand(B)));
    }
  }
}
