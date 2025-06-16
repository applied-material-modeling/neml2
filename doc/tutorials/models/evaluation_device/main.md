@insert-title:tutorials-models-evaluation-device

[TOC]

## Device

NEML2 inherits the definition of *device* from PyTorch. It is an abstraction representing the physical device where data is stored and models executed. NEML2 is primarily concerned with two types of devices: CPU and CUDA.
- **CPU** stands for Central Processing Unit, and it tranditionally handles a wide range of scientific computing tasks.
- **CUDA** functions as a programming platform that allows a GPU (Graphics Processing Unit) to act as a coprocessor to the CPU to handle specific computing tasks with massive parallelization.

NEML2 offers several high-level mechanisms for users to strategically interact with CPU and CUDA.

## Specifying a device

A device is uniquely identified by a type, which specifies the type of machine it is (e.g. CPU or CUDA GPU), and a device index or ordinal, which identifies the specific compute device when there is more than one of a certain type. The device index is optional, and in its default state represents (abstractly) "the current device". Further, there are two constraints on the value of the device index, if one is explicitly specified:
- A negative index represents the current device, a non-negative index represents a specific, concrete device.
- When the device type is CPU, the device index must be zero.

In NEML2, the device can be specified using either a string or or a predefined device type.

The string follows the schema `(cpu|cuda)[:<device-index>]`, where `cpu` or `cuda` specifies the device type, and `:<device-index>` optionally specifies a device index. For example, "cpu" represents the CPU and "cuda:1" specifies CUDA with device ID 1.

Two predefined device types are supported:
- `neml2::kCPU` which is equivalent to the string "cpu"
- `neml2::kCUDA` which is equivalent to the string "cuda" (without device ID specification)

\remark
In general, the string representation is more flexible and is universally accepted in the input file whenever a device specification is required. Parsing the string, however, inposes a small runtime overhead, and therefore it should be replaced with the corresponding device type (if applicable) in performance-critical regions of your code.

## Evaluating the model on a different device

Recall that all models take the general form of \f$ y = f(x; p, b) \f$. To invoke the forward operator \f$ f \f$ on a different device, the three ingredients \f$ x \f$, \f$ p \f$, and \f$ b \f$ must be allocated on the target device.

In our elasticity example, this breaks down to two tasks:
- Allocating the input variable \f$ \boldsymbol{\varepsilon} \f$ on the target device.
- Sending model parameters \f$ K \f$ and \f$ G \f$ to the target device.

NEML2 uses CPU as the default device. The code below demonstrates how to evaluate the model on CUDA:

<div class="tabbed">

- <b class="tab-title">C++</b>
  @source:src1
  ```cpp
  #include <torch/cuda.h>
  #include "neml2/models/Model.h"
  #include "neml2/tensors/SR2.h"

  int
  main()
  {
    using namespace neml2;
    set_default_dtype(kFloat64);
    auto model = load_model("input.i", "my_model");

    // Pick the device
    auto device = torch::cuda::is_available() ? kCUDA : kCPU;

    // Send the model parameters to the device
    model->to(device);

    // Create the strain on the device
    auto strain_name = VariableName("forces", "E");
    auto strain = SR2::fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device);

    // Evaluate the model
    auto output = model->value({{strain_name, strain}});

    // Get the stress back to CPU
    auto stress_name = VariableName("state", "S");
    auto stress = output[stress_name].to(kCPU);
  }
  ```
  @endsource
- <b class="tab-title">Python</b>
  @source:src2
  ```python
  import torch
  import neml2
  from neml2.tensors import SR2

  torch.set_default_dtype(torch.double)
  model = neml2.load_model("input.i", "my_model")

  # Pick the device
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  # Send the model parameters to the device
  model.to(device=device)

  # Create the strain on the device
  strain = SR2.fill(0.1, 0.05, -0.03, 0.02, 0.06, 0.03, device=device)

  # Evaluate the model
  output = model.value({"forces/E": strain})

  # Get the stress back to CPU
  stress = output["state/S"].to(device=torch.device("cpu"))
  ```
  @endsource

</div>

@insert-page-navigation
