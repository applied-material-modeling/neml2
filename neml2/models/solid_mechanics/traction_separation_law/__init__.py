# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Native solid-mechanics traction-separation-law models.

One file per C++ header under
``include/neml2/models/solid_mechanics/traction_separation_law/``; this
package re-imports each so the ``@register_neml2_object`` side effects fire on
package import.

Model Composition Architecture:
-------------------------------
Cohesive traction-separation models are split into three decoupled components:
1. **Damage envelope**: Evaluates rate-independent trial damage
   (:class:`BilinearDamage`, :class:`SalehaniIraniDamage`).
2. **Damage update**: Applies irreversibility capping against `damage~1` and optional
   viscous regularization (:class:`BackwardEulerViscousDamage`, :class:`ExponentialViscousDamage`).
3. **Traction assembly**: Assembles the traction vector from current damage and separation jumps
   (:class:`BilinearTraction`, :class:`SalehaniIraniTraction`, :class:`OrthotropicLinearTraction`).

Downstream Migration / Rewiring Guide:
--------------------------------------
Existing input files that passed ``critical_separation``, ``full_separation``, or ``viscosity``
directly to ``BilinearTraction`` or ``SalehaniIraniTraction`` should rewire through a
:class:`~neml2.models.common.ComposedModel`:

.. code-block:: ini

    [Models]
      [envelope]
        type = BilinearDamage
        critical_separation = ...
        full_separation = ...
      []
      [update]
        type = BackwardEulerViscousDamage
        viscosity = ...  # optional, default = 0
      []
      [traction]
        type = BilinearTraction
        penalty_stiffness = ...
      []
      [model]
        type = ComposedModel
        models = 'envelope update traction'
        additional_outputs = 'damage'
      []
"""

from .BackwardEulerViscousDamage import BackwardEulerViscousDamage
from .BenzeggaghKenaneFullSeparation import BenzeggaghKenaneFullSeparation
from .BilinearDamage import BilinearDamage
from .BilinearTraction import BilinearTraction
from .CamanhoDavilaCriticalSeparation import CamanhoDavilaCriticalSeparation
from .CavityGrowthDamageRate import CavityGrowthDamageRate
from .CavityNucleationDamageRate import CavityNucleationDamageRate
from .CavityNucleationRate import CavityNucleationRate
from .CavityVolumeOpening import CavityVolumeOpening
from .DiffusiveCavityGrowthRate import DiffusiveCavityGrowthRate
from .EffectiveCavityFraction import EffectiveCavityFraction
from .ExponentialViscousDamage import ExponentialViscousDamage
from .GrainBoundarySlidingRate import GrainBoundarySlidingRate
from .ModeMixity import ModeMixity
from .OrthotropicLinearTraction import OrthotropicLinearTraction
from .PowerLawFullSeparation import PowerLawFullSeparation
from .SalehaniIraniDamage import SalehaniIraniDamage
from .SalehaniIraniTraction import SalehaniIraniTraction
from .ShamNeedlemanTraction import ShamNeedlemanTraction
from .SlidingDrivenStrainRate import SlidingDrivenStrainRate
from .ViscousDamageUpdate import ViscousDamageUpdate

__all__ = [
    "BackwardEulerViscousDamage",
    "BenzeggaghKenaneFullSeparation",
    "BilinearDamage",
    "BilinearTraction",
    "CamanhoDavilaCriticalSeparation",
    "CavityGrowthDamageRate",
    "CavityNucleationDamageRate",
    "CavityNucleationRate",
    "CavityVolumeOpening",
    "DiffusiveCavityGrowthRate",
    "EffectiveCavityFraction",
    "ExponentialViscousDamage",
    "GrainBoundarySlidingRate",
    "ModeMixity",
    "OrthotropicLinearTraction",
    "PowerLawFullSeparation",
    "SalehaniIraniDamage",
    "SalehaniIraniTraction",
    "ShamNeedlemanTraction",
    "SlidingDrivenStrainRate",
    "ViscousDamageUpdate",
]
