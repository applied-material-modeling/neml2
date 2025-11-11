#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/vdot.h"
#include "neml2/tensors/functions/sum.h"
#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("vdot", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape1 = test::generate_tensor_shape<Vec>();
  auto shape2 = test::generate_tensor_shape<Vec>();

  DYNAMIC_SECTION(cfg.desc() << " LHS: " << shape1.desc() << " RHS: " << shape2.desc())
  {
    auto a = test::generate_random_tensor<Vec>(cfg, shape1);
    auto b = test::generate_random_tensor<Vec>(cfg, shape2);

    // NEML2 vdot
    auto c_neml2 = neml2::vdot(a, b);

    // reference implementation
    auto a_flat = a.base_flatten();
    auto b_flat = b.base_flatten();
    auto c_ref = base_sum(a_flat * b_flat);

    REQUIRE(c_neml2.base_dim() == 0); // scalar output
    REQUIRE(c_ref.base_dim() == 0);
    REQUIRE(c_neml2.dynamic_sizes() ==
            utils::broadcast_sizes(shape1.dynamic_sizes, shape2.dynamic_sizes));
    REQUIRE(c_neml2.intmd_sizes() ==
            utils::broadcast_sizes(shape1.intmd_sizes, shape2.intmd_sizes));

    REQUIRE_THAT(c_neml2, test::allclose(c_ref));
  }
}
