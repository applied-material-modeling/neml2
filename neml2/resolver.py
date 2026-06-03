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

"""DependencyResolver — topological sort of :class:`~neml2.model.Model` nodes.

Python equivalent of the C++ ``DependencyResolver<Model*, VariableName>``
template in ``include/neml2/models/DependencyResolver.h``. Variable matching
is exact string equality, mirroring the C++ behaviour. Cycle detection and
duplicate-provider validation are both enforced.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from neml2.types import TensorWrapper

if TYPE_CHECKING:
    from .model import Model


class DependencyResolver:
    """Resolve execution order for a collection of :class:`~neml2.model.Model` nodes.

    A model A is a dependency of model B if A's ``provided_items`` intersects
    B's ``consumed_items``. The resolver enforces that each variable is
    provided by at most one model and that the dependency graph is acyclic,
    then returns the models sorted in execution order (dependencies first) via
    Kahn's BFS algorithm.
    """

    def __init__(self) -> None:
        self._nodes: list[Model] = []

    def add_node(self, model: Model) -> None:
        """Register a model as a node in the dependency graph."""
        self._nodes.append(model)

    def resolve(self) -> list[Model]:
        """Return models sorted so every dependency runs before its consumer.

        Raises
        ------
        ValueError
            If two models provide the same variable name, or if a cycle exists.
        """
        # Map variable name → the unique model that provides it.
        provided_by: dict[str, Model] = {}
        for m in self._nodes:
            for name in m.provided_items:
                if name in provided_by:
                    raise ValueError(
                        f"Variable {name!r} is provided by more than one model: "
                        f"{type(provided_by[name]).__name__} and {type(m).__name__}"
                    )
                provided_by[name] = m

        # Build edges: for each (provider → consumer) pair, add a directed edge.
        in_degree: dict[Model, int] = {m: 0 for m in self._nodes}
        successors: dict[Model, list[Model]] = defaultdict(list)
        for m in self._nodes:
            seen_providers: set[Model] = set()
            for name in m.consumed_items:
                if name in provided_by:
                    provider = provided_by[name]
                    if provider is not m and provider not in seen_providers:
                        successors[provider].append(m)
                        in_degree[m] += 1
                        seen_providers.add(provider)

        # Kahn's BFS — deterministic because insertion order is preserved.
        queue: list[Model] = [m for m in self._nodes if in_degree[m] == 0]
        order: list[Model] = []
        while queue:
            m = queue.pop(0)
            order.append(m)
            for s in successors[m]:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)

        if len(order) != len(self._nodes):
            raise ValueError(
                "Circular dependency detected among models: "
                + ", ".join(type(m).__name__ for m in self._nodes if m not in order)
            )

        return order

    def inbound_items(self, order: list[Model]) -> dict[str, type[TensorWrapper]]:
        """Variables consumed by at least one model but provided by none.

        These become the inputs of the composed model. Returned as an ordered
        dict preserving first-encounter order across the execution plan.
        """
        provided: set[str] = {n for m in order for n in m.provided_items}
        result: dict[str, type[TensorWrapper]] = {}
        for m in order:
            for name, type_cls in m.input_spec.items():
                if name not in provided and name not in result:
                    result[name] = type_cls
        return result

    def outbound_items(
        self,
        order: list[Model],
        additional_outputs: list[str] | None = None,
    ) -> dict[str, type[TensorWrapper]]:
        """Variables to expose as the composed model's outputs.

        Default: all provided variables not consumed by any model (dangling
        outputs). ``additional_outputs`` forces extra intermediate variables to
        also be included in the composed model's output.
        """
        consumed: set[str] = {n for m in order for n in m.consumed_items}
        all_provided: dict[str, type[TensorWrapper]] = {}
        for m in order:
            all_provided.update(m.output_spec)

        result: dict[str, type[TensorWrapper]] = {
            name: t for name, t in all_provided.items() if name not in consumed
        }
        for name in additional_outputs or []:
            if name not in result and name in all_provided:
                result[name] = all_provided[name]

        return result
