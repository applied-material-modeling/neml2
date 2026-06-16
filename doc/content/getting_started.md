(getting-started)=
# Getting started

Once NEML2 is installed, the immediate question is

> How do I evaluate a NEML2 material model?

There are three day-to-day ways to do it — the one you reach for depends
on whether you're exploring interactively, scripting from the shell, or
deploying a calibrated model into production. All three operate on the
same artifact: a [HIT](https://github.com/applied-material-modeling/neml2-hit)
input file that names one or more models. A minimal example, used throughout this
page, lives at `tutorials/models/running_your_first_model/input.i`:

```{literalinclude} tutorials/models/running_your_first_model/input.i
:language: ini
:caption: input.i
```

## Python: `neml2.load_model`

The interactive path. `pip install neml2`, then load the model with one
call and evaluate it like any `torch.nn.Module`:

```python
import torch
import neml2
from neml2.types import SR2

model = neml2.load_model("input.i", "elasticity")
stress = model(SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0))
```

`load_model` parses the file and returns the named model ready to call.
Inputs and outputs are typed tensor wrappers (`Scalar`, `SR2`, `R2`, …)
from `neml2.types`; `Scalar(<number>)` accepts a plain Python number
directly. The full walkthrough — what each line does, batched evaluation,
reading parameters — is in
[](tutorials-models-running-your-first-model).

## CLI: `neml2-run` and `neml2-inspect`

The shell path. The Python wheel registers a handful of console scripts
for driving models without writing any Python. Two are essential for
day-to-day use:

- `neml2-inspect <input.i> <model>` prints the resolved input/output
  graph, parameters, and buffers — the first thing to reach for when
  composing or debugging a wired-up input file.
- `neml2-run <input.i> <driver>` instantiates a `Driver` and steps it
  through a load history. Useful in shell scripts, CI jobs, and
  parameter sweeps.

A `neml2-run` invocation against the bundled CLI example
(a `TransientDriver` wrapping the same elasticity model over five steps):

```bash
neml2-run input.i driver
```

`neml2-run` itself emits no output on success and returns exit code 0,
so a clean run looks empty in CI. (Drivers can still print their own
progress.) The full tool-by-tool reference, including `neml2-syntax`
for browsing the registered-object catalog, is in [](cli-utilities).

## AOTI: `neml2-compile`

The deployment path. Once a model is locked in, `neml2-compile` lowers
it through `torch.export` + AOT-Inductor into a self-contained artifact
set: one or more `.pt2` graphs, a metadata sidecar, and a drop-in HIT
stub. The result loads quickly, skips the HIT parser entirely on each
call, and is consumable from either Python or C++ without a NEML2
Python dependency:

```bash
neml2-compile input.i --model elasticity
```

This drops `elasticity.pt2`, `elasticity_jvp.pt2`, `elasticity_meta.json`,
and a drop-in HIT stub `elasticity_aoti.i` under `aoti/elasticity/`. The
end-to-end round trip — loading the artifact, runtime parameter
promotion, the trade-offs against eager mode — is covered in
[](tutorials-models-compiled). Background on the package format itself
lives in [](aoti-packages).

## Where to go next

- [](tutorials) — the end-to-end walkthroughs. The Models section is
  the canonical entry point and follows the same input file used above.
- [](tutorials-models-running-your-first-model) — line-by-line of the
  Python load + evaluate snippet, plus where typed tensors fit in.
- [](cli-utilities) — full reference for `neml2-run`, `neml2-inspect`,
  `neml2-syntax`, and `neml2-compile`, including the trailing-token HIT
  override convention.
- [](tutorials-models-compiled) — the compiled-model round trip:
  loading `.pt2` artifacts, mutable parameters across the AOTI
  boundary, and when eager beats compiled (and vice versa).
