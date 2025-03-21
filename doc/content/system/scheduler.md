# Scheduler {#system-schedulers}

[TOC]

## Heterogeneous computing

Modern computing architectures are typically heterogenous: the computing power on a node is divided across multiple asymmetric computing devices.  A typical example is a cluster where each node includes one or more CPUs, providing compute with multiple threads that all share memory, and one or more GPUs, which may or may not share memory with each other and the CPUs.  NEML2 provides software infastructure to efficiency divide, send, recover, and merge batched constitutive model updates across multiple asymmetric compute devices.

\note
Currently NEML2 only supports CPU and CUDA compute devices.  Additional accelerators supported by pytorch including Apple [mps](https://pytorch.org/docs/stable/mps.html) and Intel [xpu](https://pytorch.org/docs/stable/xpu.html) may also work, but are not currently officially supported.

NEML2 manages work scheduling, dispatch, and joining with two types of objects: dispatchers and schedulers.  These objects divide up work along a single batch axis by subdividing that axis into smaller chunks (subbatches) to run on a given device.

## Work dispatchers

NEML2 currently only has a single type of work dispatcer.  The dispatcher maintains a main thread which schedules and distributes work according the algorithm provided by the selected [work scheduler](user-content-Work-schedulers).  The main thread dispatches the work to a thread pool that maintains a specific thread for each device.  The thread picks up the task (representing a batch of work), sends the work to the device, and returns the work, once completed, to the main thread.

Setting up the work dispatcher requires providing it with lambda functions describing how to:
1. Actually run a subbatch of work on a device
2. Reduce the final collection of subbatches into a single result
3. Preprocess a subbatch before running it (for example, send it to the device)
4. Postprocess a subbatch before joining (for example, send it back to the cpu)
5. Initialize the device thread.

NEML2 provides helper routines for assembling these lambdas for models returning the constitutive model evaluate and the first derivatives of the results with respect to the inputs.

## Work schedulers

A work scheduler determines how to divide the full batch of model updates into smaller subbatches and assign those to different compute devices.  NEML2 provides several types of schedulers for a combination of scheduling models:
1. A single CPU task has exclusive access to a single compute device.
2. A single CPU task has exclusive access to one or more compute devices.
3. A set of CPU tasks shares access to one or more compute devices (this is work in progress and not available in the current release of NEML2).

A CPU task here can either mean a single programming calling NEML2, for example the runner program provided with the NEML2 release, or a single MPI rank in a distributed parallel application.  The third type of scheduler covers the most general heterogenous compute architecture, where a collection of local CPU threads have shared access to a collection of accelerator devices.  However, the first two types of schedulers can also be used in distributed computing environments, with the caveat that each task/MPI rank must have *exclusive* access to a collection of accelerators.

### SimpleScheduler

The [SimpleScheduler](#simplescheduler) is the simplest option.  It dispatches work from a single instance of NEML2 to a single device.  The user provides:
1. The torch-compatible name of the target device, e.g. `"cuda:0"`
2. A device subbatch size, which is the amount of work the scheduler will consider sending to the device at once.
3. Optionally, a device capacity.  The dispatcher will continue to send subbatches to the device until it is at capacity, then wait for it to complete some work before sending additional subbatches.  The default capacity is the subbatch size, meaning the dispatcher will send a single subbatch and wait for the device to finish before sending another.

The animation below illustrates how the simple scheduler sends work to a device.
![simple scheduler animation](animation.gif "Animation of the simple scheduler")

This animation shows the dispatcher sending subbatches to the device until it reaches capacity.  The dispatcher then blocks until the device completes a subbatch, at which it sends another subbatch to fill the device back to capacity.  This pattern of work continues until the scheduler and dispatcher complete the entire, original batch of work.

This scheduling algorithm is useful:
1. To determine the optimal subbatch size for a single device.  Typically there is an optimal subbatch size for a given model.  The size will depend on the model, the delay in sending data to the device, and the device throughput.  In practice it's best to determine that optimal size empircally by a numerical experiment and then reuse the optimal batch size in the more sophisticated scheduling algorithms.
2. In cases where each NEML2 instance, for example each MPI rank, only has exclusive access to a single accelerator.  For example, running one MPI rank per node and each node has a single GPU.

### SimpleMPIScheduler


### StaticHybridScheduler