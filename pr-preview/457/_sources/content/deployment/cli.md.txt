(cli-utilities)=
# CLI utilities

The command line is the third way to use neml2 — alongside the Python
([](python-integration)) and C++ ([](external-project-integration)) runtimes,
you can drive a model end to end from a shell, with no code. The NEML2 wheel
installs five console scripts:

| Tool            | Purpose                                              |
| :-------------- | :--------------------------------------------------- |
| `neml2-run`     | Drive a model through a load history from the shell. |
| `neml2-inspect` | Print the structural summary of a model.             |
| `neml2-syntax`  | Browse the registered-object catalog.                |
| `neml2-compile` | Export a model to an AOT-Inductor package.           |
| `neml2-stub`    | Regenerate `.pyi` type stubs for the pybind11 extension modules. |

The first four share a common style — each takes an input file (where one is
needed), each forwards trailing positional tokens as HIT overrides
(`Models/elasticity/E:=210000`), each is implemented in pure Python
under [`neml2/cli/`](https://github.com/applied-material-modeling/neml2/tree/main/neml2/cli)
and is also importable as a regular Python function.

All four of those tools accept a cumulative `--load PATH` flag for importing
user-defined extensions (custom `Model` / `Driver` / `Solver` /
`[Tensors]` / `[Data]` classes) before the tool's own work begins.
`PATH` is either a path to a `.py` file or a package directory, or a
dotted module name on `sys.path`. The matching `@register_neml2_object`
decorators fire on import and the new types become resolvable from the
input file and visible to `neml2-syntax`. Repeat `--load` for several
extensions — they import in the order given, so later modules may
depend on names registered by earlier ones.

```bash
# A custom model defined in ./my_models.py is now driveable by neml2-run
# (and inspectable by neml2-inspect / neml2-compile, listed by neml2-syntax).
neml2-run --load ./my_models.py input.i my_driver

# Multiple extensions in priority order — later may depend on earlier:
neml2-syntax --load ./shared.py --load ./project_models.py --summary --json -
```

See [](tutorials-extension-input-files) for the author's side of the
contract.

Most of the worked examples below reuse a single input file — a linear
isotropic elastic model named `elasticity`, wrapped in a five-step
`TransientDriver` named `driver`:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

## `neml2-run`

Loads an input file, instantiates the named driver, and calls its
`run()` method. Use it whenever you want to step a model through a load
history from the shell — from a shell script, a CI job, a parameter
sweep — without writing any Python.

```{program-output} neml2-run --help
```

Running our shared example: `neml2-run` itself emits no output on
success and returns a zero exit code, so a clean run looks empty in
CI. (Drivers can still print their own progress.)

```{program-output} neml2-run input.i driver
:cwd: .
:prompt: true
```

The trailing-token convention lets you tweak scalar options without
editing the file — anything after the two declared positionals is
forwarded to the HIT parser as an override. The same convention works
for `neml2-inspect` and `neml2-compile`:

```{program-output} neml2-run input.i driver Drivers/driver/save_as:=other.pt
:cwd: .
:prompt: true
```

List-valued overrides (e.g. swapping `coefficients = '200e3 0.3'` for
a different pair) need careful shell quoting around the whole value.
If that gets fiddly, edit the input file directly or use the Python
API.

(cli-neml2-inspect)=
## `neml2-inspect`

Loads a single model from an input file and prints its structural
summary: input variables, output variables, parameters, and buffers
(each with dtype, device, and shape). Use it any time you want to
verify the wiring of a model before driving it — wrong variable
names, missing parameters, and unexpected shapes all show up here in
one screenful.

```{program-output} neml2-inspect --help
```

The summary view (default):

```{program-output} neml2-inspect input.i elasticity
:cwd: .
:prompt: true
```

The same data is available as a JSON document for programmatic
consumption via `--json`:

```{program-output} neml2-inspect --json input.i elasticity
:cwd: .
:prompt: true
:ellipsis: 18
```

## `neml2-syntax`

Browses the registered-object catalog: every concrete class that can
appear in a HIT input file, along with its section, docstring, and
full option schema. It is the discovery tool — use it when you don't
yet know which class to instantiate, or when you want to see what
options a class accepts without opening the source.

```{program-output} neml2-syntax --help
```

`--summary` drops the per-option schema and emits one entry per
registered type — perfect for browsing. Combine with `--section` to
restrict to a single input-file section:

```{program-output} neml2-syntax --section Drivers --summary --json -
:cwd: .
:prompt: true
:ellipsis: 16
```

Drop `--summary` and use `--type <Name>` for the full schema of one
class — the same data the documentation's syntax catalog under
[](syntax-catalog) is generated from:

```{program-output} neml2-syntax --type LinearIsotropicElasticity --json -
:cwd: .
:prompt: true
:ellipsis: 16
```

## `neml2-compile`

Exports a model to an AOT-Inductor package — one or more `.pt2`
files plus a metadata JSON and a drop-in HIT stub — that can be
loaded later from either Python or C++ without parsing the original
input file. Use it when you want a dependency-light artifact for
inference: shipping a calibrated model to a downstream simulator,
dropping the HIT parser from a deployment image, baking parameters
into a frozen graph for performance.

```{program-output} neml2-compile --help
```

Derivative graphs are opt-in: with no `-d` flag only `forward` is
compiled and `jvp` / `jacobian` raise at runtime. Request the pairs you
need with `-d OUT:IN` (e.g. `-d stress:strain`, or `-d :` for all).

### Boundary renaming

A downstream consumer may need the compiled model's variables to carry its own
names. Rename them at the artifact's boundary with `--rename-input ORIG:NEW`,
`--rename-output ORIG:NEW`, and `--rename-parameter ORIG:NEW` (each repeatable).
The rename is **shallow**: only the names reported at the interface change — the
inputs / outputs a caller passes and reads, the promoted-parameter names in
`named_parameters()`, and the keys of every `forward` / `jvp` / `jacobian`
result. The compiled graphs and all internal wiring keep the original authored
names, so a rename is a pure relabel with identical values. `ORIG` must be an
existing structural input, output, or promoted parameter (`--rename-parameter`
targets a name from `-p`), and the resulting names must stay unique.

Renaming applies to the compiled routes only (Python `neml2.aoti.Model`, C++
`neml2::aoti::Model`, and the dispatcher); the eager routes run the un-exported
model and keep the authored names. It is available with `--model` only — with
`--driver` the bundled driver is wired to the original names, so combining them
is rejected.

The end-to-end walkthrough — emitted file layout, loading the
artifact from Python, parameter promotion (`-p <name>`), and the
trade-offs against eager mode — lives in
[](tutorials-models-compiled).

### Interactive mode (`-i` / `--interactive`)

`neml2-compile`'s flag surface is wide and interdependent: the promotable
parameter names (`-p`), the valid `OUT:IN` derivative pairs (`-d`), and the
variables available to rename (`--rename-*`) only exist once the model is loaded
and introspected. Pass `-i` / `--interactive` to be guided through the options
with prompts populated from the model, ending on a review you can run, edit, or
copy as a plain `neml2-compile` command:

```bash
neml2-compile -i input.i
```

The input file is required (give it on the command line so the shell completes
the path); `--load` extensions are honored. Interactive mode is powered by
[Questionary](https://questionary.readthedocs.io/), a lightweight dependency
carried in the `dev` extra rather than the core install — if it is missing the
flag prints a `pip install questionary` hint and exits.

## `neml2-stub`

Regenerates `.pyi` type stubs for every pybind11 extension module in the
installed `neml2` package (currently `neml2.aoti._aoti`). The stubs are
written next to their `.so` files so that pyright and IDE autocompletion
resolve `from ._aoti import Model` cleanly.

```{program-output} neml2-stub --help
```

This tool is primarily used by CI (between install and pyright) and by the
wheel-building pipeline. Any extra arguments are forwarded verbatim to
`pybind11_stubgen`.

## Where to go next

- The five scripts are also importable as `neml2.cli.run.main`,
  `neml2.cli.inspect.main`, `neml2.cli.syntax.main`,
  `neml2.cli.aoti_compile.main`, and `neml2.cli.stub.main` if you want to
  drive them in-process rather than via the shell.
- The Python-first path — `neml2.load_model(...)` + direct call —
  is the subject of [](tutorials-models-running-your-first-model)
  and is usually what you want when you need to inspect or
  post-process results in memory.
- [](tutorials-models-compiled) picks up where `neml2-compile` leaves
  off: loading `.pt2` artifacts, runtime parameter promotion, and the
  C++ deployment story.
