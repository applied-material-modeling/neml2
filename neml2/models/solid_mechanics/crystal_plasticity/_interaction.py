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

"""Shared validation for slip-system interaction-matrix models."""

from __future__ import annotations

from ....types import Scalar


def check_interaction_matrix(owner: str, state: Scalar, h: Scalar, state_name: str) -> None:
    """The interaction matrix ``h`` must be square and its slip count must match the
    per-slip ``state``, giving a clear error instead of a raw broadcast failure. Skipped
    under tracing, where the shapes are already fixed/validated eagerly.

    ``owner`` is the calling model's name and ``state_name`` the HIT option carrying the
    per-slip state, so the message names the two inputs the user wired.
    """
    import torch  # local: only for the tracing guard

    if torch.compiler.is_compiling():
        return
    hs = tuple(int(s) for s in h.shape)
    if len(hs) < 2 or hs[-1] != hs[-2]:
        raise ValueError(
            f"{owner}: interaction_matrix must be square (n_slip x n_slip); "
            f"got trailing shape {hs[-2:] if len(hs) >= 2 else hs}."
        )
    n_state = int(state.shape[-1]) if state.ndim >= 1 else 1
    if hs[-1] != n_state:
        raise ValueError(
            f"{owner}: interaction_matrix slip count {hs[-1]} does not match "
            f"{state_name} slip count {n_state}."
        )


__all__ = ["check_interaction_matrix"]
