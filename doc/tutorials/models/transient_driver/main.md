@insert-title:tutorials-models-transient-driver

[TOC]

## Problem description

In the previous tutorial, we demonstrated the use of [ImplicitUpdate](#implicitupdate) to perform constitutive update by solving an implicit system of equations. In particular, we have addressed the constitutive update in the following form
\f{align}
  \mathbf{s}_{n+1} = f(\mathbf{f}_{n+1}, \mathbf{s}_n, \mathbf{f}_n; \mathbf{p}). \label{form1}
\f}
In other words, the constitutive update takes a recursive form: Given the state and external force of the system (\f$\mathbf{s}_n\f$ and \f$\mathbf{f}_n\f$) at time step \f$n\f$, as well as the "forces" \f$\mathbf{f}_{n+1}\f$ driving the system to advance to the next step \f$n+1\f$, the model yields the state of the system for the next step \f$\mathbf{s}_{n+1}\f$.

This form of constitutive update is oftentimes sufficient when coupling with external PDE solvers: The PDE solver calculates the external driving force \f$\mathbf{f}_{n+1}\f$ and asks NEML2 to advance the state of system \f$\mathbf{s}_{n+1}\f$.

However, in many other applications such as parameter calibration, it is favorable to let NEML2 drive the constitutive update (recursively) to effectively simulate the transient response of the material. The corresponding initial-value problem can be formally written as: 

Given initial conditions \f$\mathbf{s}_0\f$ and \f$\mathbf{f}_0\f$, find, \f$\forall n \in [0, N-1]\f$,
\f{align}
  \mathbf{s}_{n+1} = f(\mathbf{f}_{n+1}, \mathbf{s}_n, \mathbf{f}_n; \mathbf{p}), \label{form2}
\f}
where \f$N\f$ is the total number of steps used to discretize the "transient".

NEML2 models, including those composed from submodels, can only be used to describe the first form of constitutive \f$\eqref{form1}\f$, i.e., a single step constitutive update. In order to simulate the initial-value problem defined in \f$\eqref{form2}\f$, a neml2::TransientDriver can be used to perform the recursive constitutive update to obtain the transient response.

## Simulating the stress-strain curve

Reusing the viscoplasticity model defined in the previous tutorial, along with the Newton-Raphson solver, a predefined driver [SDTSolidMechanicsDriver](#sdtsolidmechanicsdriver) can be used to obtain the stress-strain curve.

```
[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
    predictor = LINEAR_EXTRAPOLATION
    save_as = 'result.pt'
  []
[]

[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1
    nstep = 20
  []
  [exx]
    type = FullScalar
    value = 0.01
  []
  [eyy]
    type = FullScalar
    value = -0.005
  []
  [ezz]
    type = FullScalar
    value = -0.005
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 20
  []
[]
```

The [SDTSolidMechanicsDriver](#sdtsolidmechanicsdriver) performs the recursive constitutive update using strain control, suitable for solid mechanics models using the small strain formulation. It asks the user to specify "prescribed_time" and "prescribed_strain" which define the discretization of the transient. Two predictors are implemented for all transient drivers:
- PREVIOUS_STATE, using the converged solution from the previous step as the initial guess for the current solve;
- LINEAR_EXTRAPOLATION, linearly extrapolates the converged solutions from the previous two steps as the initial guess for the current solve.

The optional option "save_as" tells the driver to write the results into a file named "result.pt".

The following C++ code retrieves and executes the driver to obtain the stress-strain curve.
@source:src1
```cpp
#include "neml2/base/Factory.h"
#include "neml2/drivers/Driver.h"

using namespace neml2;

int
main()
{
  load_input("input.i");
  auto & driver = get_driver("driver");
  driver.run();
}
```
@endsource

\note
The Python binding for neml2::Driver does not yet exist. Instead, a separate Python package named pyzag can be used in conjunction with NEML2 to perform recursive constitutive updates in a much more efficient manner compared to the NEML2 drivers, which loop over each point in the requested time series sequentially.

The results saved in "result.pt" can be used as a regular pickled [TorchScript](https://pytorch.org/docs/stable/jit.html), which can be loaded using `torch.jit.load`. For example, the following Python script retrieves and plots the strain and stress values.
@source:src2
```python
import torch
from matplotlib import pyplot as plt

# Load the result
res = torch.jit.load("result.pt")
I = dict(res.input.named_buffers())
O = dict(res.output.named_buffers())

# Strain and stress
nstep = 20
strain = [I["{}.forces/E".format(i)][0].item() for i in range(nstep)]
stress = [O["{}.state/S".format(i)][0].item() for i in range(nstep)]

# Plot
plt.plot(strain, stress, "ko-")
plt.xlabel("Strain")
plt.ylabel("Stress")
plt.grid()
plt.tight_layout()
plt.savefig("curve.svg")
```
@endsource

![The stress-strain curve](tutorials/models/transient_driver/curve.svg){html: width=75%}

@insert-page-navigation
