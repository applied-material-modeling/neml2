(model-compilation)=
# Model compilation

NEML2 ships two ways to evaluate a model:

1. Eager Python through `neml2.load_model` — the authoring path, used
   by tutorials, tests, and pyzag training.
2. **Compiled** through `neml2-compile` — a frozen artifact (an AOT-
   Inductor `.pt2` package plus a metadata sidecar) that loads in
   milliseconds from Python or C++, with no Python source needed at
   load time.

```{image} ../../asset/neml2_compile_pipeline.svg
:alt: neml2-compile pipeline: HIT input through the export adapter to the four-file AOTI artifact set.
:align: center
:width: 95%
```

This section covers the second path, split into the developer-facing
*how it's built* and the consumer-facing *what's on disk*:

```{toctree}
:maxdepth: 1

pipeline
aoti_packages
```

If you only want to compile and load a model, the end-to-end how-to
lives in [](tutorials-models-compiled).
