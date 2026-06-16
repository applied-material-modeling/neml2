(tutorials-optimization)=
# Parameter calibration

The previous tutorials use NEML2 models in a feed-forward setting — given
parameters and inputs, compute outputs. The other major use case is
*calibration*: given input/output observations, find the parameters that
best fit them.

These tutorials cover the gradient-based calibration workflow that NEML2
supports through PyTorch autograd, plus the `pyzag` adapter for
recurrent (time-stepped) calibration.

```{toctree}
:maxdepth: 1

automatic_differentiation/main
parameter_calibration/main
pyzag
deterministic/main
statistical/main
```
