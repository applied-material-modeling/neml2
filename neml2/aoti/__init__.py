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

"""Python interface to the thin C++ ``neml2::aoti::Model`` runtime.

This module exposes :class:`Model` -- a wrapper around the bare C++ class
that loads AOTI-exported NEML2 model artifacts (a shared ``metadata.json`` plus
per-``<device>/<dtype>/`` ``.pt2`` binaries) produced by ``neml2-compile``.

The runtime exposes ``forward``, ``jvp``, ``jacobian`` plus the
parameter-derivative pair ``param_jacobian`` / ``param_vjp`` (``d(output)/
d(parameter)`` for promoted parameters) -- all keyed by the structural
input/output names recorded in the metadata. A ``named_parameters()`` dict
exposes parameters that were explicitly promoted via ``neml2-compile
--parameter NAME`` at compile time; mutating those tensors in place is
reflected on the next call. The derivative graphs are emitted only for the
``(output, input)`` / ``(output, parameter)`` pairs requested with
``neml2-compile --derivative OUT:IN`` (``jvp`` / ``jacobian`` /
``param_jacobian`` / ``param_vjp`` raise for pairs that were not compiled in).

Example usage::

    from neml2.aoti import Model

    # Pass the artifact ROOT folder; device/dtype default to the ambient
    # torch.get_default_device() / torch.get_default_dtype().
    m = Model("aoti/my_model")
    outputs = m.forward({"strain": strain_tensor})
    outputs, J = m.jacobian({"strain": strain_tensor})

    # If `--parameter E` was passed at compile time:
    m.named_parameters()["E"].fill_(210000.0)
"""

from ._aoti import ConvergenceError, Model
from ._shim import AOTIModel  # noqa: F401 (registers AOTIModel with native factory)

__all__ = ["Model", "AOTIModel", "ConvergenceError"]
