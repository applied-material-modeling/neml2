(cli-utilities)=
# CLI utilities

The NEML2 wheel installs four console scripts:

| Tool            | Purpose                                              |
| :-------------- | :--------------------------------------------------- |
| `neml2-run`     | Drive a model through a load history from the shell. |
| `neml2-inspect` | Print the structural summary of a model.             |
| `neml2-syntax`  | Browse the registered-object catalog.                |
| `neml2-compile` | Export a model to an AOT-Inductor package.           |

They share a common style — each takes an input file (where one is
needed), each forwards trailing positional tokens as HIT overrides
(`Models/elasticity/E:=210000`), each is implemented in pure Python
under [`neml2/cli/`](https://github.com/applied-material-modeling/neml2/tree/main/neml2/cli)
and is also importable as a regular Python function.

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

Running our shared example: `neml2-run` is silent on success and
returns a zero exit code, which is what makes it well-behaved inside a
CI script.

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
a different pair) involve quoting that varies by shell — either edit
the input file directly or use the Python API for those.

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

Exports a model to an AOT-Inductor package — a pair of `.pt2` files
plus a metadata JSON — that can be loaded later from either Python or
C++ without parsing the original input file. Use it when you want a
portable, dependency-light artifact for inference: shipping a
calibrated model to a downstream simulator, dropping the HIT parser
from a deployment image, baking parameters into a frozen graph for
performance.

```{program-output} neml2-compile --help
```

The end-to-end walkthrough — emitted file layout, loading the
artifact from Python, parameter promotion (`-p <name>`), and the
trade-offs against eager mode — lives in
[](tutorials-models-compiled).

## Where to go next

- The four scripts are also importable as `neml2.cli.run.main`,
  `neml2.cli.inspect.main`, `neml2.cli.syntax.main`, and
  `neml2.cli.aoti_compile.main` if you want to drive them in-process
  rather than via the shell.
- The Python-first path — `neml2.load_model(...)` + direct call —
  is the subject of [](tutorials-models-running-your-first-model)
  and is usually what you want when you need to inspect or
  post-process results in memory.
- [](tutorials-models-compiled) picks up where `neml2-compile` leaves
  off: loading `.pt2` artifacts, runtime parameter promotion, and the
  C++ deployment story.
