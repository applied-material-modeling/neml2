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

"""Python-native HIT input file factory.

Mirrors the C++ Factory / Registry pattern for Python-native models:

- :func:`register_native` — decorator that registers a model class under its
  C++ type name. ``Model`` subclasses usually inherit schema-backed
  ``from_hit(node, factory)``; non-model objects and special cases implement
  their own.
- :func:`load_input` — parse a HIT file and return a lazy :class:`_NativeInputFile`
  factory object.
- :func:`load_model` — convenience wrapper: ``load_input(path).get_model(name)``.

The factory is "dumb" — it dispatches only by type name.  Each registered class
is responsible for reading its own parameters and resolving sub-object
dependencies via the factory's ``get_model`` / ``get_solver`` /
``get_equation_system`` methods (mirroring ``NEML2Object::get_model`` etc.).
"""

from __future__ import annotations

import keyword
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import nmhit
import torch

_registry: dict[str, type] = {}

_T = TypeVar("_T")


def _check_python_attr_name(name: str, *, kind: str, owner: str) -> None:
    """Refuse names that won't survive becoming a Python attribute on an ``nn.Module``.

    Eager use only requires the name be a valid Python identifier — ``setattr`` and
    ``__getattr__`` happily handle reserved keywords like ``yield`` or ``class``.
    The trap is :func:`torch.export`: during ``GraphModule.recompile()`` torch
    rewrites the module hierarchy back into literal Python *source* (``self.X.Y.Z``).
    If any path component is a Python keyword, the parser rejects the source with
    ``SyntaxError`` and AOTI compilation collapses with a deeply nested traceback.

    We refuse the name up front so the error surfaces at HIT-load time with a
    message the user can act on, instead of at export time. Pure Python-eager use
    is also blocked — that's intentional: keeping an export-friendly name now is
    cheaper than discovering the conflict the first time someone runs
    ``neml2-compile``.
    """
    if keyword.iskeyword(name):
        raise ValueError(
            f"{kind} name {name!r} ({owner}) is a Python reserved keyword. Pick a "
            f"different name — Python keywords cannot appear as path components in "
            f"torch.export's generated forward source, so AOTI compilation will fail "
            f"at recompile time with a SyntaxError on this path."
        )


def register_native(type_name: str) -> Callable[[type[_T]], type[_T]]:
    """Decorator: register a Model (or solver/system) class under *type_name*.

    The class must provide a ``from_hit(cls, node, factory)`` classmethod that
    constructs an instance from the given nmhit Section node. ``Model``
    subclasses inherit a default implementation backed by ``HitSchema``; other
    registered object types implement it directly. The *factory* argument is a
    :class:`_NativeInputFile` instance whose ``get_model``, ``get_solver``, and
    ``get_equation_system`` methods may be called to resolve named sub-objects
    — exactly as C++ ``NEML2Object::get_model()`` does.
    """

    def decorator(cls: type[_T]) -> type[_T]:
        if not hasattr(cls, "from_hit"):
            raise TypeError(
                f"Cannot register {cls.__name__!r}: "
                "it must implement a from_hit(cls, node, factory) classmethod."
            )
        _registry[type_name] = cls
        cls._native_type_name = type_name  # type: ignore[attr-defined]
        return cls

    return decorator


class _NativeInputFile:
    """Lazy factory built from a parsed nmhit AST.

    Mirrors ``neml2::Factory``. Objects are cached by ``(section, name)`` so
    that repeated lookups return the same instance.

    Every unregistered type raises ``KeyError`` — the previous
    An internal
    Python-native object surface is now expected to cover every HIT type a
    Python caller might load; gaps surface immediately rather than silently
    routing through the C++ Python bindings.
    """

    def __init__(self, root: nmhit.Root, path: Path) -> None:
        self._root = root
        self._path = path
        # (section_plural, name) → built object
        self._cache: dict[tuple[str, str], Any] = {}
        # names currently being evaluated — used to detect circular cross-references
        self._evaluating: set[str] = set()

    # ── section lookup ────────────────────────────────────────────────────────

    def _find_in_section(self, section: str, name: str) -> nmhit.Node | None:
        """Search *all* top-level blocks named *section* for a child named *name*.

        nmhit does not automatically merge repeated explicit `[section]` blocks,
        so we must scan all of them.
        """
        for top in self._root.children(nmhit.NodeType.Section):
            if top.path() == section:
                node = top.find(name)
                if node is not None:
                    return node
        return None

    # ── generic dispatcher ────────────────────────────────────────────────────

    def _get_object(self, section: str, name: str) -> Any:
        key = (section, name)
        if key in self._cache:
            return self._cache[key]

        node = self._find_in_section(section, name)
        if node is None:
            raise KeyError(f"No [{section}/{name}] found in {self._path}")

        type_name = node.param_str("type")

        if type_name not in _registry:
            raise KeyError(
                f"Type {type_name!r} is not registered in NativeRegistry "
                f"(section={section!r}, name={name!r}). The Python-native "
                "object surface must cover every HIT type loaded through "
                "neml2; port the type or remove the [{section}/{name}] "
                "block from the input."
            )

        cls = _registry[type_name]
        # Block names cross the user→Python boundary here; if the name is a
        # Python keyword it survives setattr() / hasattr() fine but breaks
        # torch.export's GraphModule.recompile (see _check_python_attr_name).
        # Refuse early so the failure is loud and local instead of a SyntaxError
        # deep inside the AOTI lowering.
        _check_python_attr_name(name, kind="HIT block", owner=f"[{section}/{name}]")
        obj = cls.from_hit(node, self)
        # Variable-name resolution (HIT override → schema default → option name)
        # happens inside _store_schema_values via the input()/output() / var_*
        # schema fields, so there is no longer a boundary-wrapping pass here.
        #
        # Stash the HIT block name on the constructed object so downstream
        # consumers (notably ComposedModel) can register children under
        # readable HIT names instead of opaque _child_N indices. Falls back
        # silently if the object's class blocks attribute assignment (frozen
        # dataclass etc.) — composed models built directly from Python still
        # work, they just lose the HIT-name overlay.
        try:
            obj._hit_name = name
        except (AttributeError, TypeError):
            pass
        self._cache[key] = obj
        return obj

    # ── public get_* API (mirrors NEML2Object::get_model / get_solver / …) ───

    def get_model(self, name: str) -> Any:
        """Build (or return cached) the model named *name* from ``[Models]``.

        Raises ``KeyError`` if the type isn't registered in ``NativeRegistry``
        (D-054 — no fallback to C++).
        """
        return self._get_object("Models", name)

    def has_model(self, name: str) -> bool:
        """Whether a ``[Models/<name>]`` entry exists (does not build it)."""
        return self._find_in_section("Models", name) is not None

    def get_solver(self, name: str) -> Any:
        """Build (or return cached) the solver named *name* from ``[Solvers]``."""
        return self._get_object("Solvers", name)

    def get_equation_system(self, name: str) -> Any:
        """Build (or return cached) the system from ``[EquationSystems]``."""
        return self._get_object("EquationSystems", name)

    def get_data(self, name: str) -> Any:
        """Build (or return cached) the data object from ``[Data]``.

        ``[Data]`` blocks are construction-time inputs shared by reference
        across consumers — currently ``CrystalGeometry`` / ``CubicCrystal``
        for crystal-plasticity Schmid tensor lookup. No C++ fallback:
        ``[Data]`` is small, self-contained, and natively registered types
        cover everything  needs.
        """
        return self._get_object("Data", name)

    def get_driver(self, name: str) -> Any:
        """Build (or return cached) the driver from ``[Drivers]``.

        ``[Drivers]`` blocks are top-level workflow objects (e.g.
        ``TransientDriver`` for the regression suite, ``ModelUnitTest`` for
        the unit-test suite). The C++ side wires them through Factory; we
        mirror that here so a ``.i`` file's ``[Drivers]`` blocks resolve to
        Python-native classes.
        """
        return self._get_object("Drivers", name)

    # ── [Tensors] support (mode 2 of declare_typed_parameter) ────────────────

    def get_tensor(self, name: str) -> Any:
        """Evaluate a named ``[Tensors/<name>]`` Python expression and return the result.

        The block must declare ``type = Python`` and an ``expr`` parameter containing
        a Python expression (or a multi-line code block that assigns to ``result``).
        The expression is evaluated in a namespace pre-populated with ``torch``, all
        ``neml2.types`` symbols (``Scalar``, ``SR2``, ``SSR4``, free functions),
        ``math``, ``np`` (numpy, if importable), and ``tensor(name)`` for cross-references.

        Returns the raw value produced by the expression — either a ``torch.Tensor``
        or a ``TensorWrapper`` subclass.  The call site (``declare_typed_parameter`` mode 2)
        is responsible for wrapping a raw ``torch.Tensor`` into the appropriate typed wrapper.

        Raises ``KeyError`` if no ``[Tensors/<name>]`` block exists.
        """
        key = ("Tensors", name)
        if key in self._cache:
            return self._cache[key]

        node = self._find_in_section("Tensors", name)
        if node is None:
            raise KeyError(f"No [Tensors/{name}] found in {self._path}")

        type_str = node.param_str("type")

        if name in self._evaluating:
            raise RecursionError(
                f"[Tensors/{name}] cross-references itself (directly or indirectly)."
            )
        self._evaluating.add(name)
        try:
            if type_str == "Python":
                # Inline Python expression — evaluated in the typed-tensor namespace.
                result = _eval_tensor_code(
                    node.param_str("expr"), name, _build_tensor_eval_namespace(self)
                )
            elif type_str in _registry:
                # Registered tensor type (e.g. CSVScalar, CSVSR2). The class
                # builds its own typed wrapper from the HIT block.
                cls = _registry[type_str]
                result = cls.from_hit(node, self)
            else:
                raise ValueError(
                    f"[Tensors/{name}] has type={type_str!r}; expected 'Python' or a "
                    "registered tensor type. For an inline Python expression use:\n"
                    f"  [Tensors/{name}]\n"
                    "    type = Python\n"
                    "    expr = '...pytorch expression...'\n"
                    "  []"
                )
        finally:
            self._evaluating.discard(name)

        # Sub-batch tagging happens inside ``expr`` via method chaining on the
        # returned wrapper — ``Scalar.linspace(0, 1, 5).sub_batch.retag(1)``.
        # That keeps a single source of truth for the tensor pipeline (and lets
        # users compose ``.sub_batch.expand_at(...)``, ``.sub_batch.diagonalize()``
        # etc. without a parallel HIT option for each).

        self._cache[key] = result
        return result


# ── [Tensors] Python-expression helpers ───────────────────────────────────────


class _TensorNamespace(dict):
    """eval/exec namespace that lazily resolves unknown names as tensor cross-references.

    When an identifier is not found in the pre-populated symbols (``torch``,
    ``neml2.types`` exports, ``math``, ``np``), Python's name-lookup
    calls ``__missing__``.  We intercept that and try ``factory.get_tensor(key)``
    so users can write ``base`` in an expression instead of ``tensor('base')``,
    avoiding HIT's restriction that quotes may not appear inside quoted strings.

    Cycle detection is handled by ``_NativeInputFile.get_tensor`` itself.
    """

    def __init__(self, factory: _NativeInputFile, base_ns: dict) -> None:
        super().__init__(base_ns)
        self._factory = factory

    def __missing__(self, key: str) -> Any:
        try:
            val = self._factory.get_tensor(key)
        except KeyError:
            raise NameError(f"name {key!r} is not defined") from None
        self[key] = val  # cache resolved value in the namespace
        return val


def _build_tensor_eval_namespace(factory: _NativeInputFile) -> _TensorNamespace:
    """Construct the eval namespace for ``[Tensors]`` Python expressions.

    Exposes the entire ``neml2.types`` public namespace (all names listed
    in its ``__all__``) so that new types or free functions added there are
    automatically available without touching this file.

    Cross-references to other ``[Tensors]`` entries are resolved implicitly: an
    unknown identifier is looked up as ``factory.get_tensor(name)``, so users
    write ``base`` in the expression rather than ``tensor('base')`` (which would
    conflict with HIT's restriction on quotes inside quoted strings).
    """
    from . import types as _types_mod

    base_ns: dict[str, Any] = {name: getattr(_types_mod, name) for name in _types_mod.__all__}
    base_ns.update(
        {
            "torch": torch,
            "math": __import__("math"),
        }
    )
    try:
        base_ns["np"] = __import__("numpy")
    except ModuleNotFoundError:
        pass
    return _TensorNamespace(factory, base_ns)


def _dedent_hit_expr(code: str) -> str:
    """Dedent a multi-line expression string as returned by ``nmhit.param_str``.

    nmhit strips the leading whitespace from the first line of a quoted value
    (the newline and spaces between the opening quote and the first content
    character) but leaves subsequent lines intact.  Standard ``textwrap.dedent``
    therefore sees the first line with 0 indent and cannot strip anything.

    This helper deduces the common indentation from lines 2 onwards and strips
    it from the entire block, leaving relative indentation inside the code
    unchanged.
    """
    lines = code.splitlines()
    if len(lines) <= 1:
        return code.strip()
    # Lines 2+ still carry the original HIT indentation.
    rest_nonempty = [ln for ln in lines[1:] if ln.strip()]
    if not rest_nonempty:
        return lines[0].strip()
    min_indent = min(len(ln) - len(ln.lstrip()) for ln in rest_nonempty)
    dedented = [lines[0]]  # first line already has no leading whitespace
    for ln in lines[1:]:
        dedented.append(ln[min_indent:] if ln.strip() else "")
    return "\n".join(dedented)


def _eval_tensor_code(code: str, name: str, ns: dict) -> Any:
    """Eval a single-expression or exec a multi-line block; return the result."""
    src = _dedent_hit_expr(code)
    filename = f"<[Tensors/{name}]>"
    try:
        return eval(compile(src, filename, "eval"), ns)  # noqa: S307
    except SyntaxError:
        pass
    except RecursionError:
        raise
    except Exception as exc:
        raise ValueError(f"[Tensors/{name}] expr raised {type(exc).__name__}: {exc}") from exc
    try:
        exec(compile(src, filename, "exec"), ns)  # noqa: S102
    except RecursionError:
        raise
    except Exception as exc:
        raise ValueError(f"[Tensors/{name}] code raised {type(exc).__name__}: {exc}") from exc
    if "result" not in ns:
        raise ValueError(
            f"[Tensors/{name}] multi-line code block must assign its output to 'result'."
        )
    return ns["result"]


# ── public entry points ────────────────────────────────────────────────────────


def load_input(
    path: str | Path,
    *,
    pre: Sequence[str] = (),
    post: Sequence[str] = (),
    additional_args: Sequence[str] = (),
) -> _NativeInputFile:
    """Parse a HIT input file and return a lazy native factory.

    Parameters
    ----------
    path:
        Path to the HIT ``.i`` file.
    pre, post:
        Optional HIT snippets prepended / appended before parsing (same
        semantics as ``nmhit.parse_file``).
    additional_args:
        Trailing command-line HIT overrides (e.g.
        ``["Models/elasticity/E:=210000"]``). Each element is a HIT snippet
        appended after ``post`` so any ``:=`` override takes effect after
        the file's own assignments. Mirrors the C++ side's
        ``neml2::load_input(path, additional_cliargs)``.
    """
    p = Path(path)
    full_post = (*post, *additional_args)
    root = nmhit.parse_file(p, list(pre), list(full_post))
    return _NativeInputFile(root, p)


def load_string(
    text: str,
    *,
    pre: Sequence[str] = (),
    post: Sequence[str] = (),
    additional_args: Sequence[str] = (),
) -> _NativeInputFile:
    """Parse an in-memory HIT snippet and return a lazy native factory.

    Same semantics as :func:`load_input` but reads from a string (via
    ``nmhit.parse_text``) instead of a file. Used by
    :meth:`neml2.drivers.ModelUnitTest.from_string` so a unit test can
    embed its ``[Models]`` / ``[Tensors]`` / ``[Drivers]`` blocks inline.
    """
    full_post = (*post, *additional_args)
    root = nmhit.parse_text(text, list(pre), list(full_post))
    return _NativeInputFile(root, Path("<string>"))


def load_model(path: str | Path, model_name: str) -> Any:
    """Load a named model from a HIT input file as a Python-native model.

    Raises ``KeyError`` if the model's type (or any of its sub-object types)
    isn't registered in ``NativeRegistry``.

    Parameters
    ----------
    path:
        Path to the HIT ``.i`` file.
    model_name:
        Name of the model in the ``[Models]`` section.
    """
    return load_input(path).get_model(model_name)


def load_nonlinear_system(path: str | Path, name: str) -> Any:
    """Load a named system from ``[EquationSystems]`` as a Python-native object.

    Convenience wrapper around ``load_input(path).get_equation_system(name)``
    — mirrors :func:`load_model`. Returns a
    :class:`~neml2.equation_systems.NonlinearSystem` (typically a
    :class:`~neml2.equation_systems.ModelNonlinearSystem`).
    """
    return load_input(path).get_equation_system(name)


__all__ = [
    "register_native",
    "load_input",
    "load_string",
    "load_model",
    "load_nonlinear_system",
    "_NativeInputFile",
]
