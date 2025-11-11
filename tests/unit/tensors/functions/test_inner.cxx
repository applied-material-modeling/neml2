#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/inner.h"
#include "neml2/tensors/functions/sum.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEMPLATE_TEST_CASE("inner", "[tensors/functions]", FOR_ALL_TENSORBASE_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);

  // Use only floating-point dtypes (linalg_vecdot does not support integers)
  auto cfg   = test::generate_tensor_config(test::fp_dtypes());
  auto shape = test::generate_tensor_shape<TestType>();

  DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
  {
    // Generate random tensors a and b
    auto a = test::generate_random_tensor<TestType>(cfg, shape);
    auto b = test::generate_random_tensor<TestType>(cfg, shape);

    // NEML2
    auto c_neml2 = neml2::inner(a, b);

    // Ref
    auto a_flat = a.base_flatten();
    auto b_flat = b.base_flatten();
    auto c_ref = base_sum(a_flat * b_flat);


    REQUIRE(c_neml2.base_dim() == 0);
    REQUIRE(c_ref.base_dim() == 0);
    // Dynamic shape should be the broadcast of a & b
    auto dyn_expected = neml2::utils::broadcast_dynamic_sizes(std::vector<Tensor>{
        Tensor(a), Tensor(b)
    });
    REQUIRE(c_neml2.dynamic_sizes() == dyn_expected);
    REQUIRE(c_ref.dynamic_sizes() == dyn_expected);

    REQUIRE_THAT(c_neml2, test::allclose(c_ref));
  }
}
