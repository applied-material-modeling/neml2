(tutorials)=
# Tutorials

End-to-end walkthroughs of NEML2, grouped by what we're trying to do:
run and compose existing models, extend NEML2 with a model of our own,
or calibrate model parameters against data.

(tutorials-models)=
## Introduction

NEML2 ships a large collection of material models that can be used on
their own or composed into more complex ones. The wiring lives entirely
in a text-based [input file](tutorials-models-input-file), so we don't
need to write Python or C++ glue to put models together. These tutorials
walk through the everyday workflow of loading, evaluating, and composing
existing models.

```{toctree}
:maxdepth: 1
:caption: Introduction

models/input_file
models/running_your_first_model/main
models/model_parameters/main
models/evaluation_device/main
models/vectorization/main
models/cross_referencing/main
models/model_composition/main
models/model_parameters_revisited/main
models/implicit_model/main
models/transient_driver/main
models/compiled_models/main
models/compilation_internals/main
```

(tutorials-extension)=
## Extension

When the built-in library doesn't cover what we need, we can add our
own model as a small Python class deriving from `Model`, plug it into an
input file like any registered type, and optionally compile it. These
tutorials build one such model from scratch on a single running example.

```{toctree}
:maxdepth: 1
:caption: Extension

extension/argument_declaration/main
extension/connection_to_input_files/main
extension/the_forward_operator/main
extension/request_ad/main
extension/model_composition/main
```

(tutorials-optimization)=
## Parameter calibration

The earlier tutorials evaluate a model — given parameters and inputs,
compute outputs. Calibration is the inverse: given input/output
observations, find the parameters that best fit them. These tutorials
cover gradient-based calibration through PyTorch autograd, plus the
`pyzag` adapter for recurrent (time-stepped) calibration.

```{toctree}
:maxdepth: 1
:caption: Parameter calibration

optimization/automatic_differentiation/main
optimization/parameter_calibration/main
optimization/pyzag
optimization/deterministic/main
optimization/statistical/main
```
