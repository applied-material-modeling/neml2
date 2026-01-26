#include <torch/torch.h>
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/Scalar.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);

  // Number of samples
  Size ns = 2;
  // Number of strain measurements
  Size nm = 1000;

  // Elasticity tensor of the two materials
  auto youngs_modulus = Scalar::create({1e5, 2e5});
  auto poissons_ratio = Scalar::create({0.1, 0.2});
  auto C = SSR4::isotropic_E_nu(youngs_modulus, poissons_ratio);

  // (Fake) strain measurements
  auto strain = SR2::rand({nm, ns}, {}, kFloat64);

  // Perform the constitutive update
  auto stress = C * strain;

  // Do the shapes make sense?
  std::cout << "     Shape of C:" << C.batch_sizes() << C.base_sizes() << std::endl;
  std::cout << "Shape of strain: " << strain.batch_sizes() << strain.base_sizes() << std::endl;
  std::cout << "Shape of stress: " << stress.batch_sizes() << stress.base_sizes() << std::endl;
}
