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

"""neml2 -- Python-native NEML2 model authoring surface.

New constitutive models live under :mod:`neml2.models`; typed tensor
primitives (``Scalar``, ``SR2``, ``SSR4``) and free functions on them
live in :mod:`neml2.types`; the named-I/O ``Model`` base, the dependency
resolver, the chain-rule typing, the ``ComposedModel`` glue, and the
``compile_model`` AOTI export entry point all live under
:mod:`neml2.models`. Driver subclasses (``TransientDriver`` and friends)
live under :mod:`neml2.drivers`. The pyzag training adapter lives in
:mod:`neml2.pyzag`.
"""

import os
import sys
from pathlib import Path

# Windows has no rpath: the compiled extension (neml2.aoti._aoti) resolves its
# dependent DLLs -- libneml2 (neml2/lib/neml2.dll) and libtorch (torch/lib/*.dll)
# -- through the process DLL search path, not a baked-in rpath as on Linux/macOS.
# Register both directories before importing the extension below so the load
# succeeds. torch registers its own lib dir when imported, but we add it here too
# so this does not depend on torch having been imported first.
if sys.platform == "win32":  # pragma: no cover  (Windows-only; CI coverage runs on Linux)
    _neml2_lib = Path(__file__).parent / "lib"
    if _neml2_lib.is_dir():
        os.add_dll_directory(str(_neml2_lib))
    try:
        import torch as _torch

        _torch_lib = Path(_torch.__file__).parent / "lib"
        if _torch_lib.is_dir():
            os.add_dll_directory(str(_torch_lib))
    except ImportError:
        pass

from .cli import export_model_for_aoti
from .models._guard import (  # noqa: F401 (also installs the forward guard)
    allow_autograd,
    allow_einsum,
)
from .models.chain_rule import ChainRuleAction, ChainRuleDict, TangentAction

# ``compile`` intentionally shadows the builtin *in this module's namespace only*
# so the public surface reads ``neml2.compile(model)``. It is the in-process
# ``torch.compile`` accelerator (native-Python counterpart to the AOTI
# ``export_model_for_aoti`` / ``compile_model`` path), not a replacement for the
# builtin anywhere else.
from .models.compile import compile  # noqa: A004

# AOTI subpackage is gated on the optional NEML2_AOTI build. Import for side
# effects so the HIT shim registers "AOTIModel" with the factory. Two failure
# modes, treated differently:
#   * the binding was never built (no _aoti extension on disk) -- an intentional
#     lean build; stay silent.
#   * the binding is on disk but fails to import -- warn, because otherwise every
#     AOTI route goes silently missing and the only symptom is a baffling "Type
#     'AOTIModel' is not registered" KeyError when load_model later hits a
#     neml2-compile stub. The swallowed exception is the real diagnostic, so
#     surface it.
try:
    from . import aoti as _aoti_module  # noqa: F401 (registers AOTIModel)
except ImportError as _aoti_err:
    from glob import glob as _glob

    # Probe the filesystem (not an import) to tell "never built" from "unloadable".
    if _glob(str(Path(__file__).parent / "aoti" / "_aoti*.so")):
        import warnings as _warnings

        _warnings.warn(
            "NEML2's AOTI runtime binding is installed but failed to import:\n"
            f"    {type(_aoti_err).__name__}: {_aoti_err}\n"
            "The 'AOTIModel' HIT type is therefore not registered. Compiled "
            "(.pt2) models will not load -- neml2.load_model on a neml2-compile "
            "stub fails with \"Type 'AOTIModel' is not registered\", and the "
            "neml2.aoti runtime is unavailable. The exception above is the "
            "underlying cause.",
            stacklevel=2,
        )
# Eagerly expose ``neml2.pyzag`` as an attribute so notebooks / tests
# can write ``neml2.pyzag.NEML2PyzagModel(...)`` without an explicit
# import. Safe to place anywhere in this file -- ``pyzag.interface``
# imports only from the leaf modules (``neml2.es``, ``neml2.types``,
# ``neml2.models``) that are already loaded by this point. It used
# to re-export ``load_nonlinear_system`` and tripped a circular
# import; that's been removed.
from . import pyzag as pyzag  # noqa: F401
from .data import (  # noqa: F401 (registers CubicCrystal [Data] block)
    CrystalGeometry,
    CubicCrystal,
    cubic_symmetry_operators,
)
from .drivers import (  # noqa: F401 (registers TransientDriver / TransientRegression / Verification)
    TransientDriver,
    TransientRegression,
    Verification,
)
from .drivers.driver import Driver
from .drivers.ModelUnitTest import ModelUnitTest, ModelUnitTestReport
from .es import (  # noqa: F401 (side-effect: registers NonlinearSystem)
    AssembledMatrix,
    AssembledVector,
    AxisLayout,
    ModelNonlinearSystem,
    SparseMatrix,
    SparseVector,
)
from .factory import (
    load_input,
    load_model,
    load_nonlinear_system,
    load_string,
    register_neml2_object,
)
from .models import (
    chemical_reactions as _chemical_reactions_module,  # noqa: F401 (registers chemical-reactions leaves)
)
from .models import common as _common_module  # noqa: F401 (registers common leaves)
from .models import finite_volume as _finite_volume_module  # noqa: F401 (registers FV leaves)
from .models import kwn as _kwn_module  # noqa: F401 (registers KWN leaves)
from .models import (  # noqa: F401 (registers phase-field-fracture leaves)
    phase_field_fracture as _phase_field_fracture_module,
)
from .models import porous_flow as _porous_flow_module  # noqa: F401 (registers porous-flow leaves)
from .models.common import ComposedModel, ImplicitUpdate
from .models.model import Model
from .models.resolver import DependencyResolver
from .models.solid_mechanics import (  # noqa: F401 (registers solid-mechanics leaves)
    crystal_plasticity as _crystal_plasticity_module,
)
from .models.solid_mechanics import elasticity as _elasticity_module
from .models.solid_mechanics import plasticity as _plasticity_module
from .models.solid_mechanics.crystal_plasticity import (  # noqa: F401 (re-export for direct construction in tests)
    CrystalPlasticityStrainPredictor,
    ElasticStrainRate,
    OrientationRate,
    PlasticDeformationRate,
    PlasticVorticity,
    PowerLawSlipRule,
    ResolvedShear,
    SingleSlipStrengthMap,
    SumSlipRates,
    VoceSingleSlipHardeningRule,
)
from .schema import HitSchema
from .solvers import (
    DenseLU,
    Newton,
    NewtonWithLineSearch,
    NonlinearResult,
    RetCode,
    SchurComplement,
)
from .types import (  # noqa: F401 (convenience re-exports of the most common typed wrappers)
    MRP,
    R2,
    SR2,
    SSR4,
    WR2,
    MillerIndex,
    Scalar,
    Tensor,
    TensorWrapper,
    Vec,
)
from .user_tensors import (  # noqa: F401 (registers CSV<Type> [Tensors] block types)
    CSVSR2,
    CSVWR2,
    CSVScalar,
    CSVVec,
)

# Determine version
version_file = Path(__file__).parent / "version"
if version_file.exists():
    __version__ = version_file.read_text().strip()
else:
    __version__ = "unknown"

# Determine hash
hash_file = Path(__file__).parent / "hash"
if hash_file.exists():
    __hash__ = hash_file.read_text().strip()
else:
    __hash__ = "unknown"


__all__ = [
    "__version__",
    "__hash__",
    "allow_autograd",
    "allow_einsum",
    "Model",
    "DependencyResolver",
    "ComposedModel",
    "AxisLayout",
    "AssembledVector",
    "AssembledMatrix",
    "SparseVector",
    "SparseMatrix",
    "ModelNonlinearSystem",
    "DenseLU",
    "SchurComplement",
    "Newton",
    "NewtonWithLineSearch",
    "NonlinearResult",
    "RetCode",
    "ImplicitUpdate",
    "Driver",
    "TransientDriver",
    "TransientRegression",
    "CrystalGeometry",
    "CubicCrystal",
    "cubic_symmetry_operators",
    "CrystalPlasticityStrainPredictor",
    "ElasticStrainRate",
    "OrientationRate",
    "PlasticDeformationRate",
    "PlasticVorticity",
    "PowerLawSlipRule",
    "ResolvedShear",
    "SingleSlipStrengthMap",
    "SumSlipRates",
    "VoceSingleSlipHardeningRule",
    "ChainRuleDict",
    "ChainRuleAction",
    "TangentAction",
    "register_neml2_object",
    "load_input",
    "load_string",
    "load_model",
    "load_nonlinear_system",
    "export_model_for_aoti",
    "compile",
    "ModelUnitTest",
    "ModelUnitTestReport",
    "HitSchema",
]
