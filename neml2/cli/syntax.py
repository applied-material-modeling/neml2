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

"""Python-native implementation of the ``neml2-syntax`` CLI."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from ..factory import _registry
from ..schema import _MISSING, BLOCK_NAME, HitField, HitSchema
from ._extensions import add_load_argument, load_user_extensions


@dataclass(frozen=True)
class SyntaxRecord:
    type_name: str
    section: str
    doc: str
    source_path: str
    class_name: str
    hit: HitSchema | None


def _section_for(type_name: str, cls: type) -> str:
    """Read the registered class's HIT section.

    Each registered class declares its section through a ``SECTION`` class
    attribute, usually inherited from its base (:class:`neml2.model.Model`,
    :class:`neml2.driver.Driver`, :class:`neml2.es.LinearSystem`,
    ...). Classes outside an inheritance chain (linear/nonlinear solvers,
    standalone tensor/data classes) set their own.

    Falls back to ``""`` when no section is declared — surfaced by
    :func:`collect_records` validation.
    """
    del type_name
    section = getattr(cls, "SECTION", "")
    return section if isinstance(section, str) else ""


def _source_path(cls: type) -> str:
    module = cls.__module__
    prefix = "neml2."
    if module == "neml2":
        return "__init__.py"
    if module.startswith(prefix):
        return module[len(prefix) :].replace(".", "/") + ".py"
    return module.replace(".", "/") + ".py"


def _ascii_doc(text: str, context: str) -> str:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError(f"{context} contains non-ASCII documentation text") from e
    return text


def collect_records() -> list[SyntaxRecord]:
    """Collect Python-native syntax records from the native registry.

    Every registered class must declare a HIT ``SECTION`` (usually inherited
    from its base class — see :func:`_section_for`). A missing section means
    the type would silently fall out of the syntax catalog, so this raises
    rather than skipping.
    """
    records: list[SyntaxRecord] = []
    for type_name, cls in sorted(_registry.items()):
        section = _section_for(type_name, cls)
        if not section:
            raise ValueError(
                f"{cls.__module__}.{cls.__qualname__} (registered as {type_name!r}) has "
                "no SECTION declared. Add ``SECTION = '<Section>'`` to the class or one "
                "of its base classes so neml2-syntax knows which HIT section it belongs "
                "to."
            )
        doc = _ascii_doc(inspect.getdoc(cls) or "", f"{type_name} doc")
        hit = getattr(cls, "hit", None)
        records.append(
            SyntaxRecord(
                type_name=type_name,
                section=section,
                doc=doc,
                source_path=_source_path(cls),
                class_name="",
                hit=hit if isinstance(hit, HitSchema) else None,
            )
        )
    return records


def _ftype(field: HitField) -> str:
    if field.kind in {"input", "var_inputs"}:
        return "INPUT"
    if field.kind == "output":
        return "OUTPUT"
    if field.kind in {"parameter", "parameters"}:
        return "PARAMETER"
    return "NONE"


def _type_name(value_type: Any) -> str:
    if isinstance(value_type, type):
        return value_type.__name__
    if value_type is None:
        return ""
    return str(value_type)


def _default_value(field: HitField) -> str:
    default = field.default
    if default is _MISSING or default is None or default is BLOCK_NAME:
        return ""
    if isinstance(default, (list, tuple)):
        return " ".join(str(v) for v in default)
    return str(default)


def _field_to_json(field: HitField) -> dict[str, Any]:
    return {
        "name": field.name,
        "doc": _ascii_doc(field.doc, f"{field.name} option doc"),
        "ftype": _ftype(field),
        "required": field.required,
        "type": _type_name(field.value_type),
        "value": _default_value(field),
    }


def record_to_json(record: SyntaxRecord, *, include_options: bool = True) -> dict[str, Any]:
    j: dict[str, Any] = {
        "type": record.type_name,
        "section": record.section,
        "doc": record.doc,
        "source_path": record.source_path,
        "class_name": record.class_name,
    }
    if include_options and record.hit is not None:
        # ``derived_*`` fields reference another option to derive a variable
        # name; they aren't user-facing HIT options of their own and would
        # confuse the syntax catalog by duplicating the base option's entry.
        j["options"] = [
            _field_to_json(field)
            for field in record.hit.fields
            if field.kind not in {"derived_input", "derived_output"}
        ]
    return j


def _summary_json(record: SyntaxRecord) -> dict[str, Any]:
    return record_to_json(record, include_options=False)


def _matching_records(
    records: list[SyntaxRecord], *, section_filter: str = "", type_filter: str = ""
) -> list[SyntaxRecord]:
    return [
        r
        for r in records
        if (not section_filter or r.section == section_filter)
        and (not type_filter or r.type_name == type_filter)
    ]


def _arg_used(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neml2-syntax",
        description="Extract Python-native object syntax from the native registry as JSON.",
    )
    parser.add_argument("--json", default="syntax.json", help="redirect JSON output to a file")
    parser.add_argument(
        "--section",
        default="",
        help="only emit objects whose input-file section matches",
    )
    parser.add_argument(
        "--type",
        default="",
        help="only emit the object whose registered type matches",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="emit only type, section, source_path, and doc string for each object",
    )
    parser.add_argument(
        "--server", action="store_true", help="run as a JSON server on stdin/stdout"
    )
    add_load_argument(parser)
    return parser


def _run_server(records: list[SyntaxRecord], stdin: TextIO, stdout: TextIO) -> int:
    for line in stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps({"id": None, "error": "parse error"}) + "\n")
            stdout.flush()
            continue

        req_id = req.get("id")
        method = req.get("method", "")
        if method == "list_sections":
            result = sorted({r.section for r in records if r.section})
        elif method == "list_types":
            section = req.get("section", "")
            result = [_summary_json(r) for r in _matching_records(records, section_filter=section)]
        elif method == "get_options":
            type_name = req.get("type", "")
            result = None
            for record in records:
                if record.type_name == type_name:
                    result = record_to_json(record, include_options=True)
                    break
        else:
            stdout.write(json.dumps({"id": req_id, "error": "unknown method"}) + "\n")
            stdout.flush()
            continue

        stdout.write(json.dumps({"id": req_id, "result": result}) + "\n")
        stdout.flush()
    return 0


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    in_stream: TextIO = sys.stdin if stdin is None else stdin
    out_stream: TextIO = sys.stdout if stdout is None else stdout
    err_stream: TextIO = sys.stderr if stderr is None else stderr

    parser = _build_parser()
    args = parser.parse_args(args_list)

    if args.server and any(
        _arg_used(args_list, name) for name in ("--json", "--section", "--type", "--summary")
    ):
        err_stream.write(
            "error: --server is incompatible with --json, --section, --type, and --summary\n"
        )
        return 1

    try:
        load_user_extensions(args.load)
    except ImportError as exc:
        err_stream.write(f"error: {exc}\n")
        return 1

    records = collect_records()

    if args.server:
        return _run_server(records, in_stream, out_stream)

    result = [
        record_to_json(r, include_options=not args.summary)
        for r in _matching_records(records, section_filter=args.section, type_filter=args.type)
    ]

    text = json.dumps(result, indent=2) + "\n"
    if args.json == "-":
        out_stream.write(text)
    else:
        Path(args.json).write_text(text, encoding="utf-8")
    return 0


__all__ = ["SyntaxRecord", "collect_records", "main", "record_to_json"]


if __name__ == "__main__":
    raise SystemExit(main())
