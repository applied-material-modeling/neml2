import torch
from neml2.tensors import Scalar, SR2, SSR4

# Number of samples
ns = 2
# Number of strain measurements
nm = 1000

# Elasticity tensor of the two materials
youngs_modulus = Scalar(torch.tensor([1e5, 2e5], dtype=torch.float64))
poissons_ratio = Scalar(torch.tensor([0.1, 0.2], dtype=torch.float64))
C = SSR4.isotropic_E_nu(youngs_modulus, poissons_ratio)

# (Fake) strain measurements
strain = SR2.rand((nm, ns), (), dtype=torch.float64)

# Perform the constitutive update
stress = C * strain

# Do the shapes make sense?
print("     Shape of C:", C.batch.shape, C.base.shape)
print("Shape of strain:", strain.batch.shape, strain.base.shape)
print("Shape of stress:", stress.batch.shape, stress.base.shape)
