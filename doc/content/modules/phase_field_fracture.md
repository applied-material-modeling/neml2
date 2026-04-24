# Phase-Field Fracture {#phase-field-fracture}

[TOC]

The **phase-field fracture** physics module is a collection of objects serving as building blocks for composing models describing the evolution of regularized fracture, and the accompanying loss of materials' load-bearing capacity. The models defined in this module offer a set of phase-field fracture constitutive choices that can be utilized to set up a fracture simulation using both staggered and monolithic solution schemes.

This module can be coupled with other physics modules to describe energy dissipation coupled with fracture. For example, the **solid mechanics** module can be coupled to simulate a variety of (visco)plastic dissipation. This documentation offers a short description of each object under this module followed by a comprehensive example showing how each of them can be used to compose a bigger model describing fracture evolution locally at independent material points.

## Crack geometric functions

The crack geometric function, \f$ \alpha(\phi) \f$ determines the distribution of the crack phase field that governs the "shape" of the smeared crack. Given the phase field, \f$ \phi \f$, it calculates the corresponding scalar-valued crack geometric function. Below are some example crack geometric functions.

- [CrackGeometricFunctionAT1](#crackgeometricfunctionat1)
- [CrackGeometricFunctionAT2](#crackgeometricfunctionat2)

## Degradation functions

Generally, as the fracture evolves, the material gradually loses its load-carrying capacity. Given the phase field, \f$ \phi \f$, the degradation function, \f$ g(\phi) \f$ calculates the corresponding scalar-valued function to degrade the materials' stiffness. Some example degradation functions are listed below.

- [PowerDegradationFunction](#powerdegradationfunction)
- [RationalDegradationFunction](#rationaldegradationfunction)

## Strain Energy Density

This module also offers definitions of the strain energy density to facilitate the setup of variational constitutive update. Given the strain as input, the strain energy density objects calculate the active part of the strain energy density, \f$ \psi_\mathrm{active} \f$ that drives the fracture propagation and the inactive part, \f$ \psi_\mathrm{inactive} \f$.


## Example

In this section, we'll describe how everything discussed above fits together to compose a bigger model that can be used to simulate fracture evolution at material points. Consider the following governing equation for crack propagation

\f{align}
 f & = \eta \dot{\phi} + \frac{\partial}{\partial \phi} \left( \alpha \frac{G_c}{c_0 l} + g(\phi) \psi_\mathrm{active} \right) \ge 0, \quad \dot{\phi} \ge 0, \quad f \dot{\phi} = 0,

\f}
which can be equivalently formulated using the Fischer-Burmeister complementarity condition as

\f{align}
 f + \dot{\phi} - \sqrt{ f^2 + \dot{\phi}^2 } = 0.
\f}

The complementarity condition can be implicitly solved for the phase field, \f$ \phi \f$ using the [ImplicitUpdate](#implicitupdate) model. The implicit equation can also be coupled with other physics. The following example input file walks through the model composition where \f$ d \f$ is used instead of \f$ \phi \f$ as the phase-field variable for the sake of brevity.

@list-input:tests/regression/phase_field_fracture/elastic_brittle_fracture/small_deformation.i


The stress evaluated at the end of the input file acts only as a post-processor to verify our results.


