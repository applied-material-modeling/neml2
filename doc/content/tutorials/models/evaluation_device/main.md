@insert-title:tutorials-models-evaluation-device

[TOC]

## Device

NEML2 inherits the definition of *device* from PyTorch. It is an abstraction representing the physical device where data is stored and models executed. NEML2 is primarily concerned with two types of devices: CPU and CUDA.
- **CPU** stands for Central Processing Unit, and it tranditionally handles a wide range of scientific computing tasks.
- **CUDA** functions as a programming platform that allows a GPU (Graphics Processing Unit) to act as a coprocessor to the CPU to handle specific computing tasks with massive parallelization.

NEML2 offers several high-level mechanisms for users to strategically interact with CPU and CUDA.

\note
Other evaluation devices that are supported by PyTorch, such as XPU, HIP, MPS, etc., are also generally compatible with NEML2. Currently, NEML2 models and tensor operations are not being continuously tested on these other devices, and the support for these devices remains to be community effort.

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
  @list:cpp:evaluation_device/ex1.cxx
- <b class="tab-title">Python</b>
  @list:cpp:evaluation_device/ex2.py

</div>

@insert-page-navigation
