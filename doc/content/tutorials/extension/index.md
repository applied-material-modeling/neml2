(tutorials-extension)=
# Writing custom model

:::{seealso}
Before diving in, we recommend reading [](contributing) to
get the development environment in shape.
:::

NEML2 is extensible by design: most users who outgrow the built-in
model library add their own as a small Python class derived from
`Model`, plug it into an input file like any registered type, and
optionally compile it to AOT-Inductor for downstream consumers.

This set of tutorials walks through defining one such model from
scratch using a small running example.

```{toctree}
:maxdepth: 1

argument_declaration/main
connection_to_input_files/main
the_forward_operator/main
model_composition/main
```
