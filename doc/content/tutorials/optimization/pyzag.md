@insert-title:tutorials-optimization-pyzag

[TOC]

The [previous tutorial](@ref tutorials-optimization-parameter-calibration) demonstrates the calibration of NEML2 model parameters using a simple gradient descent algorithm. While the algorithm and workflow are largely applicable for practical model calibration tasks, several challenges (discussed below) remain to be addressed.

The NEML2 development team offers a companion Python package named [pyzag](https://github.com/applied-material-modeling/pyzag) to address these challenges. The challenges and their corresponding solutions are briefly discussed below. Two examples are available for interested users:
- `python/examples/deterministic.ipynb`: Example calibration of a deterministic NEML2 material model against stress-strain curves using pyzag.
- `python/examples/statistical.ipynb`: Example calibration of a stochastic NEML2 material model against stress-strain curves using pyzag.

## Challenge 1: Obtaining parameter derivatives of recursive nonlinear functions

Many experiments concern the "transient" response of the material. Such transient response can be obtained through time integration of NEML2 material models (see e.g., @ref tutorials-models-transient-driver). More generally speaking, the transient is obtained by solving a recursive, generally nonlinear, system of equations.

Due to the recursive nature of these material models, relying on backward-mode automatic differentiation to obtain the parameter derivatives becomes practically infeasible, especially when the number of recursive (time) steps gets large. For example, suppose we integrate the material model of a plastic material through \f$ N \f$ time steps to obtain its stress response \f$ \sigma_0, \sigma_1, \sigma_1, ..., \sigma_{N-1}, \sigma_{N} \f$, the derivative of the final stress \f$ \sigma_{N} \f$ w.r.t. the material parameter (i.e., Young's modulus) \f$ E \f$, depends on the stress at the previous step \f$\sigma_{N-1}\f$. Recursively, it also depends on the stress states at all time steps. Moreover, the solution procedure usually involves multiple Newton-Raphson iterations per time step, so the resulting function graph (needed by the AD back-propagation) is extremely deep.

When back-propagating such a deep function graph, a considerable amount of memory is required, and the required memory scales linearly with the number of time steps.

**pyzag** addresses this problem using the adjoint method. Instead of relying on the native PyTorch backward automatic differentiation, pyzag implements custom hooks to calculate the parameter derivatives by solving the adjoint problem of the original nonlinear recursive function. The memory required for solving the adjoint problem is orders of magnitudes smaller than that required by backward AD.

## Challenge 2: Data scarcity

Unlike most of the machine learning applications, experimental data available for calibrating material response is usually scarce. Among many of the consequences, the implication for computational efficiency is that GPUs are oftentimes starving for work, because
- Conventional material models are much cheaper to evaluate compared to machine learning architectures such as neural networks;
- For recursive nonlinear functions, we have to perform time integration sequentially, step-by-step.

**pyzag** addresses this challenge by observing that the statement made in the second bullet point above is wrong. It uses a novel vectorized time integration algorithm to parallelize the solution scheme over multiple time steps, effectively sending more work at a time to the device, which leads to ~10x speed up. The basic idea is documented in [this paper](https://arxiv.org/abs/2310.08649).

## Challenge 3: Uncertainty quantification

In reality, material parameters are non-deterministic, from experiment to experiment, heat to heat, etc., in addition to noise in experimental measurements. Many methods exist to quantify the uncertainty, such as Markov-Chain Monte Carlo (MCMC) and Stochastic Variational Inference (SVI). **pyzag** extends the popular **pyro** library to perform UQ.

In our experience, when the number of parameters to calibrate is large, SVI can be more efficient than MCMC. However, it is usually not a trivial task to convert an existing, deterministic model into a statistical one for the purpose of SVI. **pyzag** again addresses that challenge by providing utilities to automatically convert a deterministic NEML2 material model into a statistical one, given user-defined priors.
