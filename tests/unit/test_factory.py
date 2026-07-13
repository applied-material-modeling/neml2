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

"""Tests for the Python-native HIT factory (neml2.factory)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neml2.factory import _NativeInputFile, load_input, load_model, register_neml2_object
from neml2.models.common import ComposedModel
from neml2.models.model import Model
from neml2.models.solid_mechanics.elasticity import LinearIsotropicElasticity

# ---------------------------------------------------------------------------
# Paths to HIT test fixtures
# ---------------------------------------------------------------------------

_ELASTICITY_I = (
    Path(__file__).resolve().parents[1]
    / "models/solid_mechanics/elasticity/LinearIsotropicElasticity.i"
)


# ---------------------------------------------------------------------------
# register_neml2_object / from_hit protocol
# ---------------------------------------------------------------------------


def test_schema_less_model_requires_hit_schema_or_from_hit_override():
    import nmhit

    @register_neml2_object("_MissingHitSchema")
    class _Bad(Model):
        input_spec = {}
        output_spec = {}

        def forward(self):  # type: ignore[override]
            pass

    hit_text = """
[Models]
  [m]
    type = _MissingHitSchema
  []
[]
"""
    factory = _NativeInputFile(nmhit.parse_text(hit_text), Path("synthetic.i"))
    with pytest.raises(TypeError, match="HitSchema|override"):
        factory.get_model("m")


def test_register_neml2_object_requires_from_hit_for_non_model_classes():
    with pytest.raises(TypeError, match="from_hit"):

        @register_neml2_object("_MissingFromHit")
        class _Bad:
            pass


def test_register_neml2_object_tags_class():
    from neml2.models.solid_mechanics.elasticity import LinearIsotropicElasticity

    assert LinearIsotropicElasticity._native_type_name == "LinearIsotropicElasticity"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Leaf model: LinearIsotropicElasticity
# ---------------------------------------------------------------------------


def test_load_leaf_model_returns_native_type():
    # The fixture sets strain = 'strain' (the C++ default), so no VariableRemapper.
    model = load_model(_ELASTICITY_I, "model")
    assert isinstance(model, LinearIsotropicElasticity)


def test_load_leaf_model_has_correct_moduli():
    # E=100, nu=0.3 from LinearIsotropicElasticity.i  (coefficients = '100 0.3')
    model = load_model(_ELASTICITY_I, "model")
    assert isinstance(model, LinearIsotropicElasticity)
    assert abs(model.E.data.item() - 100.0) < 1e-10
    assert abs(model.nu.data.item() - 0.3) < 1e-10


def test_native_elasticity_matches_independent_isotropic_reference():
    """Forward output of the loaded native model matches an independent
    closed-form reference (E=100, ν=0.3 per the HIT file) to 1e-12.

    The reference is computed inline from the textbook isotropic formula
    ``σ = K·tr(ε)·I + 2G·dev(ε)`` — no dependence on the C++ bindings
    (`neml2.load_model`, `neml2.tensors.SR2`)
    delete. Any future change to the Python-native ``LinearIsotropicElasticity``
    is caught against the math, not against a parallel implementation we're
    on track to remove.
    """
    import math  # noqa: PLC0415

    from neml2.types import SR2  # noqa: PLC0415

    native = load_model(_ELASTICITY_I, "model")
    assert isinstance(native, LinearIsotropicElasticity)

    # HIT specifies E=100, ν=0.3.
    E, nu = 100.0, 0.3
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    gen = torch.Generator().manual_seed(0)
    strain_t = torch.randn(8, 6, generator=gen, dtype=torch.float64)

    # Native forward.
    native_stress = native(SR2(strain_t))
    native_arr = native_stress.data if hasattr(native_stress, "data") else native_stress

    # Independent reference in Mandel packing. The first three Mandel slots
    # are diagonal entries, the next three are sqrt(2)·{yz, xz, xy}. The
    # Mandel-weighted shear factors carry through both the trace projection
    # (zero on off-diagonals) and the deviator subtraction (linear) without
    # any extra factor of sqrt(2) on the σ side, so the formula
    # σ = K·tr(ε)·I + 2G·dev(ε) applies component-wise in Mandel.
    trace = strain_t[..., 0] + strain_t[..., 1] + strain_t[..., 2]
    I_mandel = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    vol_eps = (trace / 3.0).unsqueeze(-1) * I_mandel
    dev_eps = strain_t - vol_eps
    expected = 3.0 * K * vol_eps + 2.0 * G * dev_eps

    assert torch.allclose(native_arr, expected, rtol=1e-12, atol=1e-12), (
        f"max_abs={float((native_arr - expected).abs().max()):.3e}, "
        f"E={E}, nu={nu}, K={K:.6f}, G={G:.6f}, "
        f"sqrt(2)={math.sqrt(2.0):.6f} (Mandel shear factor)"
    )


# ---------------------------------------------------------------------------
# Unregistered types — strict KeyError, no fallback
# ---------------------------------------------------------------------------


def test_unknown_type_raises_key_error():
    """An older Loading a
    HIT model whose ``type`` isn't registered in ``NativeRegistry`` now
    raises ``KeyError`` immediately — no warning, no silent route through
    the C++ Python bindings."""
    import nmhit

    hit_text = """
[Models]
  [m]
    type = ThisModelTypeDoesNotExist
  []
[]
"""
    root = nmhit.parse_text(hit_text)
    factory = _NativeInputFile(root, Path("synthetic.i"))
    with pytest.raises(KeyError, match="ThisModelTypeDoesNotExist"):
        factory.get_model("m")


# ---------------------------------------------------------------------------
# ComposedModel from HIT
# ---------------------------------------------------------------------------


def test_load_composed_model_type():
    """ComposedModel with natively-typed children compiles correctly via
    a synthetic HIT (single registered ``LinearIsotropicElasticity`` child)."""
    hit_text = """
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [composed]
    type = ComposedModel
    models = 'elasticity'
  []
[]
"""
    import nmhit

    root = nmhit.parse_text(hit_text)
    factory = _NativeInputFile(root, Path("synthetic.i"))
    m = factory.get_model("composed")
    assert isinstance(m, ComposedModel)


def test_composed_model_forward_matches_leaf():
    from neml2.types import SR2

    hit_text = """
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [composed]
    type = ComposedModel
    models = 'elasticity'
  []
[]
"""
    import nmhit

    root = nmhit.parse_text(hit_text)
    factory = _NativeInputFile(root, Path("synthetic.i"))

    leaf = factory.get_model("elasticity")
    composed = factory.get_model("composed")

    gen = torch.Generator().manual_seed(7)
    strain = SR2(torch.randn(4, 6, generator=gen, dtype=torch.float64))

    (composed_stress,) = composed(strain.data)
    leaf_stress = leaf(strain)
    assert torch.allclose(composed_stress.data, leaf_stress.data, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Solver + EquationSystem from HIT
# ---------------------------------------------------------------------------


def test_load_newton_solver_from_hit():
    from neml2.solvers import Newton

    hit_text = """
[Solvers]
  [lu]
    type = DenseLU
  []
  [newton]
    type = Newton
    linear_solver = 'lu'
    abs_tol = 1e-8
    rel_tol = 1e-6
  []
[]
"""
    import nmhit

    root = nmhit.parse_text(hit_text)
    factory = _NativeInputFile(root, Path("synthetic.i"))
    solver = factory.get_solver("newton")
    assert isinstance(solver, Newton)
    assert abs(solver.atol - 1e-8) < 1e-15
    assert abs(solver.rtol - 1e-6) < 1e-15


def test_newton_with_linesearch_from_hit_honors_verbose():
    """Regression: ``NewtonWithLineSearch.from_hit`` must honor the inherited
    ``verbose`` option. It previously failed to read it, so a ``verbose = true``
    input silently produced a quiet solver -- the schema accepts the field
    (inherited from ``Newton``), so no error flagged the dropped option."""
    from neml2.solvers import NewtonWithLineSearch

    hit_text = """
[Solvers]
  [newton]
    type = NewtonWithLineSearch
    verbose = true
  []
[]
"""
    import nmhit

    root = nmhit.parse_text(hit_text)
    factory = _NativeInputFile(root, Path("synthetic.i"))
    solver = factory.get_solver("newton")
    assert isinstance(solver, NewtonWithLineSearch)
    assert solver.verbose is True


# ---------------------------------------------------------------------------
# load_input / load_model public API
# ---------------------------------------------------------------------------


def test_load_input_returns_native_input_file():
    f = load_input(_ELASTICITY_I)
    assert isinstance(f, _NativeInputFile)


def test_load_model_is_sugar_for_load_input_get_model():
    f = load_input(_ELASTICITY_I)
    m1 = f.get_model("model")
    m2 = load_model(_ELASTICITY_I, "model")
    assert type(m1) is type(m2)
    assert isinstance(m1, LinearIsotropicElasticity)
    assert isinstance(m2, LinearIsotropicElasticity)
    assert torch.equal(m1.E.data, m2.E.data)


def test_load_input_caches_objects():
    """Repeated get_model calls for the same name return the identical object."""
    f = load_input(_ELASTICITY_I)
    m1 = f.get_model("model")
    m2 = f.get_model("model")
    assert m1 is m2


# ---------------------------------------------------------------------------
# Multiple [Models] sections in the same file
# ---------------------------------------------------------------------------


def test_find_model_in_second_section_block():
    """Factory must scan all [Models] blocks, not just the first."""
    import nmhit

    hit_text = """
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]

[Models]
  [composed]
    type = ComposedModel
    models = 'elasticity'
  []
[]
"""
    root = nmhit.parse_text(hit_text)
    factory = _NativeInputFile(root, Path("synthetic.i"))
    # 'composed' lives in the second [Models] block; must still be findable.
    m = factory.get_model("composed")
    assert isinstance(m, ComposedModel)


# ---------------------------------------------------------------------------
# Schema-side variable rename: non-default HIT variable names
# ---------------------------------------------------------------------------


def test_variable_rename_resolved_at_construction():
    """Non-default strain name in HIT → input_spec key carries the external name.

    Path B (no VariableRemapper): the schema's input() field reads the HIT
    option of the same name and rewrites input_spec in-place; the model graph
    sees the external name directly with no boundary wrap.
    """
    import nmhit

    from neml2.types import SR2

    hit_text = """
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    strain = 'my_strain'
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
"""
    root = nmhit.parse_text(hit_text)
    factory = _NativeInputFile(root, Path("synthetic.i"))
    model = factory.get_model("elasticity")

    assert "my_strain" in model.input_spec
    assert "strain" not in model.input_spec
    assert "stress" in model.output_spec

    direct = LinearIsotropicElasticity(1e5, 0.3)
    gen = torch.Generator().manual_seed(42)
    strain_t = torch.randn(4, 6, generator=gen, dtype=torch.float64)

    renamed_stress = model(SR2(strain_t))
    direct_stress = direct(SR2(strain_t))
    assert torch.allclose(renamed_stress.data, direct_stress.data, rtol=1e-12, atol=1e-12)


def test_load_rejects_python_keyword_block_name():
    """HIT block names that are Python reserved keywords (`yield`, `class`, etc.)
    must be refused at load time. Eager Python use would silently work, but the
    same name breaks torch.export's GraphModule.recompile (it generates literal
    Python source ``self.X.yield.Y`` which the parser rejects with SyntaxError).
    See _check_python_attr_name in neml2.factory.
    """
    import nmhit
    import pytest

    hit_text = """
[Models]
  [yield]
    type = YieldFunction
    yield_stress = 1000
  []
[]
"""
    root = nmhit.parse_text(hit_text)
    factory = _NativeInputFile(root, Path("synthetic.i"))
    with pytest.raises(ValueError, match=r"Python reserved keyword"):
        factory.get_model("yield")


def test_register_typed_parameter_rejects_python_keyword_attr_name():
    """The schema-author-controlled `attr=...` for parameter/buffer fields must
    also avoid Python keywords. Same root cause: keyword attribute names break
    torch.export's source-form recompile.
    """
    import pytest
    import torch

    from neml2.models.model import Model
    from neml2.types import Scalar

    class _Bare(Model):
        input_spec = {}
        output_spec = {}

        def forward(self, *args, **kwargs):  # pragma: no cover — never called
            raise NotImplementedError

    m = _Bare()
    with pytest.raises(ValueError, match=r"Python reserved keyword"):
        m.register_typed_parameter("class", Scalar(torch.tensor(1.0)))


def test_normalize_load_target():
    """``load_model`` / ``load_input`` normalize the load-time ``device`` /
    ``dtype`` (torch objects or their string spellings) to the string form the
    ``AOTIModel`` shim + C++ ctor accept; ``None`` passes through so the ctor
    falls back to the ambient torch default downstream."""
    import torch

    from neml2.factory import _normalize_load_target

    assert _normalize_load_target(None, None) == (None, None)
    assert _normalize_load_target("cpu", "float64") == ("cpu", "float64")
    assert _normalize_load_target(torch.device("cpu"), torch.float64) == ("cpu", "float64")
    # A concrete cuda index is preserved (pins a dispatcher Model to that GPU);
    # the ``torch.`` prefix on a dtype's ``str()`` is stripped.
    assert _normalize_load_target(torch.device("cuda", 1), torch.float32) == ("cuda:1", "float32")
