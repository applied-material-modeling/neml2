//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "ComputeStressBase.h"
#include "SymmetricRankTwoTensor.h"
#include "SymmetricRankFourTensor.h"

/**
 * Feed an external NEML2 stress + Jacobian (produced by the [NEML2] block via
 * NEML2ToMOOSEMaterialProperty) into MOOSE's OLD (legacy tensor-mechanics) stress
 * framework -- the old-system analog of ComputeLagrangianObjectiveCustomSymmetricStress.
 *
 * NEML2 is fed the co-rotational mechanical strain (from ComputeFiniteStrain), so its
 * stress lives in the co-rotational frame. To match the native finite-strain path, this
 * material rotates that stress into the current configuration using the accumulated
 * rotation (rotation_increment * rotation_total_old), exactly as the native
 * ComputeFiniteStrainElasticStress / ComputeMultipleInelasticStress do.
 */
class ComputeNEML2StressOldSystem : public ComputeStressBase
{
public:
  static InputParameters validParams();

  ComputeNEML2StressOldSystem(const InputParameters & parameters);

protected:
  virtual void initQpStatefulProperties() override;
  virtual void computeQpStress() override;

  /// NEML2 (co-rotational) Cauchy stress, from the [NEML2] block
  const MaterialProperty<SymmetricRankTwoTensor> & _neml2_stress;
  /// NEML2 dstress/dstrain (symmetric fourth order)
  const MaterialProperty<SymmetricRankFourTensor> & _neml2_jacobian;
  /// mechanical strain (from the old-system strain calculator), for the elastic_strain output
  const MaterialProperty<RankTwoTensor> & _mechanical_strain;
  /// incremental rotation from the finite-strain calculator
  const MaterialProperty<RankTwoTensor> & _rotation_increment;
  /// accumulated rotation (stateful), mapping the co-rotated stress to the current config
  MaterialProperty<RankTwoTensor> & _rotation_total;
  const MaterialProperty<RankTwoTensor> & _rotation_total_old;
};
