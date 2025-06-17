# Chemical Reactions {#chemical-reactions}

[TOC]

The chemical reactions physics module is a collection of objects serving as building blocks for composing models describing various chemical reactions. The models defined in this module do not attempt to define reaction kinetics from first principles. Instead, these models define macroscale, homogenized reaction rates of some composite system, which can be coupled with other physics such as solid mechanics and porous flow.

## Reaction mechanisms

A variety of empirical reaction mechanisms are provided by the module. These models relate the degree of conversion \f$ \alpha \f$ to the reaction rate (defined as the rate of change of the degree of conversion).

## Effective volume

The macroscale chemical reaction models are often defined for a given control mass, i.e., the system under consideration is assumed to be impermeable w.r.t. the neighborhood of the material point.

However, other physics modules, such as solid mechanics or porous flow, are often derived under the assumptions of control volume. In order to couple chemical reactions to other physics, the [EffectiveVolume](#effectivevolume) model can be used to calculate the effective volume corresponding to the control mass during the reactions as a function of mass fractions of each reactant. The effective volume can then serve as the bridging quantity when coupling to other physics, such as volume-dependent eigenstrain and/or deformation Jacobian.

## Examples

- [Thermosetting pyrolysis](#thermosetting-pyrolysis)
- [Reactive infiltration](#reactive-infiltration)
