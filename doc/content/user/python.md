# Python Package {#python-package}

[TOC]

\note
The NEML2 Python package is experimental. APIs are expected to change.

## Load and evaluate a model from input file

With the NEML2 Python package, the same input file in the other [tutorial](#cpp-backend) can be directly used in a Python script. The Python APIs closely ressembles the C++ APIs. For example, the previous C++ example translates to the following Python script.

```python
import neml2
from neml2.tensors import Scalar, SR2
from neml2.math import batch_stack

model = neml2.load_model("input.i", "model")

strain1 = SR2.fill(0.1, 0.2, 0.3, -0.1, -0.1, 0.2)
strain2 = SR2.fill(0.2, 0.2, 0.1, -0.1, -0.2, -0.5)
strain3 = SR2.fill(0.3, -0.2, 0.05, -0.1, -0.3, 0.1)
strain = batch_stack([strain1, strain2, strain3])

output = model.value({"forces/E": strain})
stress = output["state/S"]
```

All is the same in the equivalent C++ example, the above Python script parses the input file named "input.i", loads the linear elasticity model named "model", constructs 3 strain tensors, and finally performs the 3 material updates simultaneously.

## Model parameters

Model parameters can be modified in a similar fashion. One notable difference is that, in Python, model parameters can be directly accessed as class attributes, i.e.,
```python
E = model.E
E.set_(Scalar.full(200.0))
E.requires_grad_()
```
Similarly, multiple parameters can be updated at once:
```python
model.set_parameters({"E": Scalar.full(200.0),
                      "nu": Scalar.full(0.25)})
```
