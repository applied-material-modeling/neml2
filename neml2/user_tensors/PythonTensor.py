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

"""``Python`` user-tensor — inline Python expression evaluated at parse time.

Previously a special case inside ``factory.get_tensor``; promoted to a
properly registered ``[Tensors]`` class so the syntax catalog documents it
alongside the ``CSV<Type>`` family.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ..factory import _build_tensor_eval_namespace, _eval_tensor_code, register_neml2_object
from ..schema import HitSchema, option

if TYPE_CHECKING:
    import nmhit

    from ..factory import _NativeInputFile


@register_neml2_object("Python")
class PythonTensor:
    """Tensor built from an inline Python expression.

    The ``expr`` option is evaluated in a namespace pre-populated with
    ``torch``, every public name from ``neml2.types`` (``Scalar``, ``SR2``,
    ``SSR4``, free functions), ``math``, and ``np`` (numpy, if importable).
    Cross-references to other ``[Tensors]`` entries resolve by bare
    identifier: writing ``base`` is equivalent to ``tensor('base')``, which
    avoids HIT's restriction on nested quotes.

    The expression's value (``torch.Tensor`` or a ``TensorWrapper`` subclass)
    is returned verbatim; the call site (typically
    :meth:`neml2.model.Model.declare_typed_parameter` mode 2) is responsible
    for wrapping a raw tensor into a typed wrapper if needed.
    """

    SECTION: ClassVar[str] = "Tensors"

    hit = HitSchema(
        option(
            "expr",
            str,
            "Python expression producing the tensor value. A multi-line block "
            "must assign its result to a variable named ``result``.",
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> Any:
        return _eval_tensor_code(
            node.param_str("expr"),
            node.path().rsplit("/", 1)[-1],
            _build_tensor_eval_namespace(factory),
        )


__all__ = ["PythonTensor"]
