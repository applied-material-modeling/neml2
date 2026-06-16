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
            If two models provide the same variable name without
            :func:`~neml2.schema.output` priority disambiguation, or if a
            cycle exists.
        """
        # Map variable name → list of providers. Multiple providers are
        # allowed as long as their declared priorities ("low" < default <
        # "high") disambiguate the order; otherwise the duplicate is an error.
        providers_by_name: dict[str, list[Model]] = defaultdict(list)
        for m in self._nodes:
            for name in m.provided_items:
                providers_by_name[name].append(m)

        # The "winning" provider per name is the highest-priority one. The
        # composed model's output_spec sources this provider's value.
        # ``priority_edges`` records the ordering implied by priority claims
        # (low → default → high) so a same-name producer chain executes in
        # the right order even when there's no data-flow edge between them.
        provided_by: dict[str, Model] = {}
        priority_edges: list[tuple[Model, Model]] = []
        for name, providers in providers_by_name.items():
            if len(providers) == 1:
                provided_by[name] = providers[0]
                continue
            buckets: dict[str | None, list[Model]] = defaultdict(list)
            for p in providers:
                buckets[getattr(p, "output_priorities", {}).get(name)].append(p)
            for level, members in buckets.items():
                if len(members) > 1:
                    label = "default" if level is None else f"priority={level!r}"
                    raise ValueError(
                        f"Variable {name!r} is provided by more than one model "
                        f"at the same {label} level: "
                        + ", ".join(type(m).__name__ for m in members)
                        + ". At most one provider per (name, priority) is allowed."
                    )
            # Resolve order: low → default → high. The highest level present
            # is the winner; chain edges enforce execution order.
            ordered: list[Model] = []
            for level in ("low", None, "high"):
                if level in buckets:
                    ordered.append(buckets[level][0])
            for a, b in zip(ordered[:-1], ordered[1:], strict=False):
                priority_edges.append((a, b))
            provided_by[name] = ordered[-1]

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
        # Priority-implied edges: low → default → high among same-name providers.
        for src, dst in priority_edges:
            if dst not in successors[src]:
                successors[src].append(dst)
                in_degree[dst] += 1

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
        """Variables that need to flow in from outside the composed model.

        A name X is inbound iff at least one consumer of X has no
        earlier-in-execution-order model that provides X. This is the
        priority-aware refinement of "consumed but not provided":

        * Regular pipeline (producer before consumer) — X has an earlier
          provider, not inbound.
        * In-place transformer with no upstream sibling (``input='X'`` /
          ``output='X'`` and no other producer of X) — X has no earlier
          provider for the only consumer, so X stays inbound. The wrapper
          can't satisfy its own input from its own output.
        * Priority chain (low → default → high all on X) — the chain's
          earlier producers satisfy the higher consumer's read; X is not
          inbound from the chain's perspective.
        """
        idx_of: dict[Model, int] = {m: i for i, m in enumerate(order)}
        provider_indices: dict[str, list[int]] = defaultdict(list)
        for m in order:
            for name in m.provided_items:
                provider_indices[name].append(idx_of[m])
        result: dict[str, type[TensorWrapper]] = {}
        for m in order:
            for name, type_cls in m.input_spec.items():
                if name in result:
                    continue
                earlier = [p for p in provider_indices.get(name, []) if p < idx_of[m]]
                if not earlier:
                    result[name] = type_cls
        return result

    def outbound_items(
        self,
        order: list[Model],
        additional_outputs: list[str] | None = None,
    ) -> dict[str, type[TensorWrapper]]:
        """Variables to expose as the composed model's outputs.

        Default: variables whose final producer in *order* (the priority
        winner when multiple models provide the same name) isn't consumed
        by any later model. ``additional_outputs`` forces extra intermediate
        variables to also be included in the composed model's output.

        The priority-aware filter is what lets an in-place transformer like
        ``FixOrientation`` (``input='orientation'`` / ``output='orientation'``)
        expose its rewritten ``orientation`` even though it also *consumes*
        ``orientation`` -- that consumption is reading the lower-priority
        sibling's value, satisfied internally by the priority chain, not an
        external consumption of the winner's output.
        """
        # Index each model's position so "later than producer" is O(1).
        idx_of: dict[Model, int] = {m: i for i, m in enumerate(order)}

        # Map each name to its final producer (winner). When a name has
        # multiple providers, the winner is the one with the largest order
        # index (priority-resolved low → default → high).
        winner_of: dict[str, Model] = {}
        for m in order:
            for name in m.provided_items:
                if name not in winner_of or idx_of[m] > idx_of[winner_of[name]]:
                    winner_of[name] = m

        all_provided: dict[str, type[TensorWrapper]] = {}
        for m in order:
            all_provided.update(m.output_spec)

        # A name is "internally consumed" iff some model strictly AFTER the
        # winner reads it. Consumption by the winner itself, or by any
        # earlier sibling in the priority chain, is the chain wiring -- not
        # an external sink.
        consumed_after_winner: set[str] = set()
        for m in order:
            for name in m.consumed_items:
                winner = winner_of.get(name)
                if winner is None:
                    continue
                if idx_of[m] > idx_of[winner]:
                    consumed_after_winner.add(name)

        result: dict[str, type[TensorWrapper]] = {
            name: t for name, t in all_provided.items() if name not in consumed_after_winner
        }
        for name in additional_outputs or []:
            if name not in result and name in all_provided:
                result[name] = all_provided[name]

        return result
