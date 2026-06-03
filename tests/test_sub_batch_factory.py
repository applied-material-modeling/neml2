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

"""Factory HIT integration — sub-batch tagging via method chaining in ``expr``.

Sub-batch metadata lives entirely in the wrapper, so the HIT ``expr`` string
is the single source of truth: users chain ``.with_sub_batch(N)`` (and other
sub-batch ops) on the result. No parallel HIT option is needed.
"""

from __future__ import annotations

import torch

from neml2.factory import load_input
from neml2.types import SR2, Scalar


def _write_hit(tmp_path, body: str):
    path = tmp_path / "input.i"
    path.write_text(body)
    return load_input(path)


# ---------- default behaviour without method-chained tagging ----------


def test_python_tensor_without_chain_has_no_sub_batch(tmp_path):
    inp = _write_hit(
        tmp_path,
        """
[Tensors]
  [bins]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 5, dtype=torch.float64))'
  []
[]
""",
    )
    t = inp.get_tensor("bins")
    assert isinstance(t, Scalar)
    assert t.sub_batch_ndim == 0
    assert t.shape == torch.Size([5])


# ---------- with_sub_batch via expr chain ----------


def test_with_sub_batch_in_expr_promotes_trailing_axis(tmp_path):
    inp = _write_hit(
        tmp_path,
        """
[Tensors]
  [bins]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 5, dtype=torch.float64)).with_sub_batch(1)'
  []
[]
""",
    )
    t = inp.get_tensor("bins")
    assert isinstance(t, Scalar)
    assert t.sub_batch_ndim == 1
    assert t.sub_batch_shape == torch.Size([5])
    assert t.dynamic_batch_shape == torch.Size([])


def test_with_sub_batch_two_dims_in_expr(tmp_path):
    inp = _write_hit(
        tmp_path,
        """
[Tensors]
  [field]
    type = Python
    expr = 'Scalar(torch.zeros(3, 4, dtype=torch.float64)).with_sub_batch(2)'
  []
[]
""",
    )
    t = inp.get_tensor("field")
    assert isinstance(t, Scalar)
    assert t.sub_batch_ndim == 2
    assert t.sub_batch_shape == torch.Size([3, 4])


def test_with_sub_batch_in_expr_on_sr2(tmp_path):
    inp = _write_hit(
        tmp_path,
        """
[Tensors]
  [stresses]
    type = Python
    expr = 'SR2(torch.zeros(7, 6, dtype=torch.float64)).with_sub_batch(1)'
  []
[]
""",
    )
    t = inp.get_tensor("stresses")
    assert isinstance(t, SR2)
    assert t.sub_batch_ndim == 1
    assert t.sub_batch_shape == torch.Size([7])
    assert t.base_shape == torch.Size([6])


# ---------- chained sub-batch ops in expr ----------


def test_with_sub_batch_then_expand_in_expr(tmp_path):
    """The whole sub-batch op surface is reachable from inside ``expr``."""
    inp = _write_hit(
        tmp_path,
        """
[Tensors]
  [tiled]
    type = Python
    expr = '''
      bins = Scalar(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
      result = bins.with_sub_batch(1).sub_batch_expand(4)
    '''
  []
[]
""",
    )
    t = inp.get_tensor("tiled")
    assert isinstance(t, Scalar)
    assert t.sub_batch_ndim == 2
    # ``sub_batch_expand`` inserts at sub-batch position 0 (the leading slot),
    # so the new size-4 axis precedes the original size-3 axis.
    assert t.sub_batch_shape == torch.Size([4, 3])
    # Every row of the new leading axis carries the same (1, 2, 3) values.
    for k in range(4):
        assert torch.equal(t.data[k], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


def test_with_sub_batch_then_diagonalize_in_expr(tmp_path):
    """Chained diagonalize materialises an (L, L) per-site block."""
    inp = _write_hit(
        tmp_path,
        """
[Tensors]
  [diag]
    type = Python
    expr = '''
      vals = Scalar(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64))
      result = vals.with_sub_batch(1).sub_batch_diagonalize()
    '''
  []
[]
""",
    )
    t = inp.get_tensor("diag")
    assert t.sub_batch_ndim == 2
    assert t.sub_batch_shape == torch.Size([3, 3])
    assert torch.equal(t.data, torch.diag(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)))


def test_with_sub_batch_chain_in_multiline_expr(tmp_path):
    """Multi-line ``'''...'''`` expr (nmhit 0.2.0+) hosts wrapped chains too.

    Python method chains need explicit parens to span lines; the chained
    expression is bound to ``result`` so :func:`_eval_tensor_code`'s
    ``exec`` fallback picks it up.
    """
    inp = _write_hit(
        tmp_path,
        """
[Tensors]
  [stack]
    type = Python
    expr = '''
      bins = Scalar(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
      result = (bins
                  .with_sub_batch(1)
                  .sub_batch_expand(4))
    '''
  []
[]
""",
    )
    t = inp.get_tensor("stack")
    assert isinstance(t, Scalar)
    assert t.sub_batch_ndim == 2
    assert t.sub_batch_shape == torch.Size([4, 3])


# ---------- end-to-end: parameter resolution preserves the chained tag ----------


def test_intermediate_tensor_resolves_through_declare_typed_parameter(tmp_path):
    """A model parameter resolved via a [Tensors] cross-ref works correctly."""
    from neml2.model import Model

    class _PerBinScale(Model):
        input_spec = {"x": Scalar}
        output_spec = {"y": Scalar}
        k: Scalar  # populated by declare_typed_parameter

        def __init__(self, k_spec, *, factory=None):
            super().__init__()
            self.declare_typed_parameter("k", k_spec, Scalar, factory=factory)

        def forward(self, *inputs, v=None):  # type: ignore[override]
            (x,) = inputs
            return Scalar(self.k.data * x.data)

    inp = _write_hit(
        tmp_path,
        """
[Tensors]
  [per_bin_k]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)).with_sub_batch(1)'
  []
[]
""",
    )
    m = _PerBinScale("per_bin_k", factory=inp)
    # The registered parameter is a flat tensor (pytree dropped sub_batch_ndim
    # at registration), but the underlying value carries the right shape so
    # broadcasting still works at forward time. This is the
    # ``drop_field_names`` contract from
    assert m.k.data.shape == torch.Size([3])
    x = Scalar(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
    y = m(x)
    assert torch.equal(y.data, torch.tensor([10.0, 40.0, 90.0], dtype=torch.float64))
