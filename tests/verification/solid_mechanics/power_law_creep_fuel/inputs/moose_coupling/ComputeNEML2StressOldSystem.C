//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ComputeNEML2StressOldSystem.h"

registerMooseObject("SolidMechanicsApp", ComputeNEML2StressOldSystem);

InputParameters
ComputeNEML2StressOldSystem::validParams()
{
  InputParameters params = ComputeStressBase::validParams();
  params.addClassDescription(
      "Bridge a NEML2 stress + Jacobian (produced by the [NEML2] block) into the OLD "
      "tensor-mechanics stress framework, rotating the co-rotational NEML2 stress into the "
      "current configuration to match the native finite-strain path.");
  params.addParam<MaterialPropertyName>(
      "neml2_stress", "neml2_stress", "NEML2 (co-rotational) Cauchy stress material property");
  params.addParam<MaterialPropertyName>(
      "neml2_jacobian",
      "dneml2_stress/dneml2_strain",
      "NEML2 dstress/dstrain material property (symmetric fourth order)");
  return params;
}

ComputeNEML2StressOldSystem::ComputeNEML2StressOldSystem(const InputParameters & parameters)
  : ComputeStressBase(parameters),
    _neml2_stress(getMaterialProperty<SymmetricRankTwoTensor>("neml2_stress")),
    _neml2_jacobian(getMaterialProperty<SymmetricRankFourTensor>("neml2_jacobian")),
    _mechanical_strain(getMaterialPropertyByName<RankTwoTensor>(_base_name + "mechanical_strain")),
    _rotation_increment(getMaterialPropertyByName<RankTwoTensor>(_base_name + "rotation_increment")),
    _rotation_total(declareProperty<RankTwoTensor>(_base_name + "neml2_rotation_total")),
    _rotation_total_old(
        getMaterialPropertyOldByName<RankTwoTensor>(_base_name + "neml2_rotation_total"))
{
}

void
ComputeNEML2StressOldSystem::initQpStatefulProperties()
{
  ComputeStressBase::initQpStatefulProperties();
  _rotation_total[_qp] = RankTwoTensor::Identity();
}

void
ComputeNEML2StressOldSystem::computeQpStress()
{
  // Accumulate the total rotation (same recurrence as ComputeFiniteStrainElasticStress).
  _rotation_total[_qp] = _rotation_increment[_qp] * _rotation_total_old[_qp];

  // NEML2 is fed the co-rotational mechanical strain, so neml2_stress lives in the
  // co-rotational frame; rotate it into the current configuration.
  const RankTwoTensor s(_neml2_stress[_qp]);
  _stress[_qp] = _rotation_total[_qp] * s * _rotation_total[_qp].transpose();

  // NEML2 consistent tangent. Rotation of the 4th-order tangent is neglected -- exactly as the
  // native finite-strain path flags its own Jacobian "NOT the exact jacobian"; at the coupled
  // tolerances (nl_rel_tol 1e-4) this does not affect convergence.
  _Jacobian_mult[_qp] = RankFourTensor(_neml2_jacobian[_qp]);

  // Elastic strain output equals the mechanical strain (elastic + creep tracked inside NEML2).
  _elastic_strain[_qp] = _mechanical_strain[_qp];
}
