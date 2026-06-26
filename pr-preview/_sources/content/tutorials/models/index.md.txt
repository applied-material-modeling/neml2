(tutorials-models)=
# Working with models

NEML2 ships with a large collection of material models. They can be used
on their own or composed into more complex models.

Once NEML2 is installed (see [Installation](../../installation/install.md)),
the wiring lives entirely in a text-based
[input file](tutorials-models-input-file) — NEML2 parses it into a
runnable model at load time, so you don't need to write Python or
C++ glue to compose models.

The tutorials in this section walk through the everyday workflow of
using existing NEML2 models.

```{toctree}
:maxdepth: 1

input_file
running_your_first_model/main
model_parameters/main
evaluation_device/main
vectorization/main
cross_referencing/main
model_composition/main
model_parameters_revisited/main
implicit_model/main
transient_driver/main
compiled_models/main
```
