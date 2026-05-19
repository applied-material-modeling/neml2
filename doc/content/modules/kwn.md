# KWN Precipitation {#kwn}

[TOC]

The KWN physics module is a collection of objects for composing Kampmann–Wagner Numerical (KWN) models of precipitation, growth, and coarsening in a multi-component matrix. KWN models track a discretized size distribution of precipitates, allowing nucleation, growth, and coarsening (Ostwald ripening) to emerge from the same population balance equation. The conventions and notation followed here are those of [Ury et al., 2023](https://doi.org/10.1016/j.actamat.2023.118988), and the SFFK driving force formulation follows [Svoboda et al., 2004](https://doi.org/10.1016/j.msea.2004.06.018).

A complete KWN model in NEML2 is composed by combining the precipitation building blocks below with the [finite volume](@ref finite-volume) module, which provides the spatial discretization of the population balance equation for the number density distribution of the precipitates as a function of their radius.

## Theory

This section uses the following indexing conventions for the size distribution of precipitates:
1. Subscripts indicate the particle size bin index in the finite volume distretization of the size distribution: \f$f_i\f$.
2. Parenthetical superscripts indicates the number of precipitates being considered: \f$f^{(j)}\f$.

and the following conventions for the chemical concentration of the varioius elements contributing to reactions:
1. Subscripts indicates the number of species in the matrix, which must be the union of all the species contributing to all the precipitate reactions: \f$x_k\f$.
2. The set of chemical species contributing to precipitate \f$(j)\f$ is \f$\mathcal{S}_j\f$.

The model can consider multiple precipitate species competing for the same resivoir of checmical species in the matrix.  The state of the system is the precipitate number density per unit volume, \f$n_i^{(j)}\f$, in each radius bin \f$\left[R^{(j)}_{i-1/2}, R^{(j)}_{i+1/2}\right]\f$ with center \f$R^{(j)}_i\f$ for each precipitate \f$(j)\f$. 

Concentrations are stored as mole fractions \f$x_k\f$. The matrix is treated as a homogeneous reservoir whose composition is determined by mass conservation against the precipitates.

The finite volume discretization assumes that the size of all the particles in a single bin is the same.

### Mass conservation

The volume fraction of precipitate \f$(j)\f$ is obtained by integrating its size distribution,

\f[
  f^{(j)} = \sum_i \frac{4}{3}\pi \left(R_i^{(j)}\right)^3 n_i^{(j)},
\f]

and the current matrix concentration of species \f$k\f$ follows from the closure that everything not locked up in precipitates must be in solution,

\f[
  x^\infty_k = \frac{x_{0,k} - \sum_j f^{(j)} x^{(j)}_k}{1 - \sum_j f^{(j)}},
\f]

where \f$x_{0,k}\f$ is the initial matrix concentration of species \f$k\f$, \f$f^{(j)}\f$ is the volume fraction of precipitate \f$(j)\f$, and \f$x^{(j)}_k\f$ is the concentration of species \f$k\f$ in that precipitate (taken as zero for \f$k \notin \mathcal{S}_j\f$).  This assumes the initial volume fraction of the precipitates is zero.

### Population balance

Each precipitate population evolves according to

\f[
  \frac{\partial n^{(j)}}{\partial t} + \frac{\partial \left(n^{(j)} \dot{R}^{(j)}\right)}{\partial R^{(j)}} = J^{(j)}_{\text{nuc}},
\f]

a 1D advection equation in radius space with a nucleation source on the right-hand side. Spatial discretization is provided by the [finite volume](@ref finite-volume) module: cells are defined on \f$R^{(j)}\f$, the growth rate \f$\dot{R}^{(j)}\f$ becomes the advection velocity at cell edges, and the nucleation source dumps the newly created precipitates in the bin containing the critical radius indicated by classical nucleation theory (but see below for some caveats). 

### Growth rate

Two growth-rate models are provided.

The simpler **rate-limited** form solves for a single rate-limiting species \f$\star \in \mathcal{S}_j\f$,

\f[
  \dot{R}^{(j)} = \frac{1}{R^{(j)}}\, D_\star\, \frac{x^\infty_\star - x^{*,(j)}_\star}{x^{(j)}_\star - x^{*,(j)}_\star},
\f]

with \f$x^{*,(j)}_\star\f$ the matrix equilibrium concentration of the rate-limiting species in equilibrium with precipitate \f$(j)\f$. The growth rate vanishes correctly as \f$x^\infty_\star \rightarrow x^{*,(j)}_\star\f$.

The more general **SFFK** form (Svoboda–Fischer–Fratzl–Kozeschnik) handles multi-component diffusion in the dilute, diffusion-limited regime,

\f[
  \dot{R}^{(j)} = \frac{1}{R^{(j)}}\, \frac{\Delta G^{(j)}_{\text{chem}} + \Delta G^{(j)}_{\text{surf}} + \Delta G^{(j)}_{\text{el}}}{R_g T \, S^{(j)}},
  \qquad
  S^{(j)} = \sum_{k \in \mathcal{S}_j} \frac{\left(\Delta x^{(j)}_k\right)^2}{D_k\, x^\infty_k},
\f]

with \f$\Delta x^{(j)}_k = x^{(j)}_k - x^{*,(j)}_k\f$ and \f$D_k\f$ the matrix diffusivity of species \f$k\f$. The driving force is split into chemical, surface, and elastic contributions:

\f[
  \Delta G^{(j)}_{\text{chem}} = \sum_{k \in \mathcal{S}_j} \Delta x^{(j)}_k \left[\mu^{\text{matrix}}_k(x^\infty) - \mu^{\text{equil},(j)}_k\right], \quad
  \Delta G^{(j)}_{\text{surf}} = -\frac{2 \gamma^{(j)} V_m^{(j)}}{R^{(j)}}, \quad
  \Delta G^{(j)}_{\text{el}} = -3 V_m^{(j)} \varepsilon_0^{(j)} \sigma_h + \frac{6 V_m^{(j)} \mu K}{3K + 4\mu}\left(\varepsilon_0^{(j)}\right)^2.
\f]

In these expressions \f$V_m^{(j)}\f$ is the molar volume of the precipitate.

The chemical driving force is the sum of the chemical potential difference between the species in the matrix \f$\mu_{k}^{\text{matrix}}\f$ as a function of the current matrix composition and the species in the precipitate \f$\mu^{\text{equil},(j)}_k\f$, assuming that the precipitate is always at equilibrium concentraiton.

The surface eenergy driving force is just the contribution of a spherical precipitate at a given size, with \f$\gamma^{(j)}\f$ the surface energy.

The \f$\Delta G^{(j)}_{\text{el}}\f$ term is the standard expression for a coherent, isotropic spherical inclusion in an isotropic matrix with misfit strain \f$\varepsilon_0^{(j)}\f$; \f$\mu\f$ and \f$K\f$ are the matrix shear and bulk moduli, and \f$\sigma_h\f$ the matrix hydrostatic stress.

### Nucleation flux

Nucleation occurs at the critical radius,

\f[
  J^{(j)}_{\text{nuc}} = Z^{(j)} \beta^{(j)} N_0^{(j)} \exp\left(-\frac{\Delta G^{*,(j)}}{k_B T}\right) \delta\left(R^{(j)} - R^{(j)}_{\text{crit}}\right),
\f]

with the classical-nucleation-theory expressions

\f[
  R^{(j)}_{\text{crit}} = \frac{2 \gamma^{(j)} V_m^{(j)}}{\Delta g^{(j)}_v}, \qquad
  \Delta G^{*,(j)} = \frac{16 \pi}{3}\, \frac{\left(\gamma^{(j)}\right)^3 \left(V_m^{(j)}\right)^2}{\left(\Delta g^{(j)}_v\right)^2},
\f]

\f[
  Z^{(j)} = \frac{V_m^{(j)}}{2\pi N_a \left(R^{(j)}_{\text{crit}}\right)^2}\sqrt{\frac{\gamma^{(j)}}{k_B T}}, \qquad
  \beta^{(j)} = \frac{4 \pi \left(R^{(j)}_{\text{crit}}\right)^2 N_a^{4/3}}{\left(V_m^{(j)}\right)^{4/3}}\, \frac{1}{S^{(j)}}.
\f]

\f$\gamma^{(j)}\f$ is the precipitate surface energy, \f$V_m^{(j)}\f$ its molar volume, \f$\Delta g^{(j)}_v\f$ the volumetric driving force for nucleation of precipitate \f$(j)\f$ (in J/mol — the per-mole convention used throughout `NucleationBarrierAndCriticalRadius`), \f$N_0^{(j)}\f$ the nucleation site density, \f$N_a\f$ Avogadro's number, \f$k_B\f$ the Boltzmann constant, and \f$S^{(j)}\f$ the same projected diffusivity sum that appears in the SFFK growth rate.

The volumetric driving force \f$\Delta g^{(j)}_v\f$ can be supplied either as a CALPHAD-derived tabulation against composition and temperature (the pattern used by the Al-Cu example below) or assembled from the participating species' matrix and equilibrium concentrations with `IdealSolutionVolumetricDrivingForce`, which evaluates the Hu--Cocks ideal-solution form \f$\Delta g_v = RT \sum_k w_k \ln(c_k/c_k^{\text{eq}})\f$. Setting every weight to 1 recovers the "product of all components" convention used for compound precipitates.

The critical radius \f$R^{(j)}_{\text{crit}}\f$ is the minimum stable size of a precipitate in the classical theory, i.e. when a stable precipitate can nucleate because the chemical driving force overcomes the surface energy requird to form a (spherical) precipitate.

***However***, for most systems \f$R_{\text{crit}}\f$ will be much smaller than the smallest cell size for a reasonably-sized finite volume discretization.  Therefore, in the examples, the Dirac delta at the critical radius is approximated discretely by `DumpInSmallestBin`, which deposits the flux magnitude into the smallest bin. 

## Building blocks

The KWN module exposes the following objects, each implementing one of the equations above. They are designed to be composed together (typically inside a `ComposedModel` driven by `ImplicitUpdate`) rather than used in isolation.

| Object                                | Role                                                                                       |
| :------------------------------------ | :----------------------------------------------------------------------------------------- |
| `PrecipitateVolumeFraction`           | Sums \f$\tfrac{4}{3}\pi (R_i^{(j)})^3 n_i^{(j)}\f$ over bins to obtain \f$f^{(j)}\f$       |
| `CurrentConcentration`                | Mass-balance closure for \f$x^\infty_k\f$ given the precipitate populations                |
| `ProjectedDiffusivitySum`             | Computes the multi-species sum \f$S^{(j)}\f$ shared by SFFK and nucleation                 |
| `ChemicalGibbsFreeEnergyDifference`   | Assembles \f$\Delta G^{(j)}_{\text{chem}}\f$ from per-species potentials                   |
| `IdealSolutionVolumetricDrivingForce` | Assembles the molar driving force \f$RT \sum_k w_k \ln(c_k/c_k^{\text{eq}})\f$ from concentrations |
| `RateLimitedPrecipitateGrowthRate`    | Single-species, equilibrium-driven growth rate \f$\dot{R}^{(j)}\f$                         |
| `SFFKPrecipitationGrowthRate`         | SFFK multi-component growth rate \f$\dot{R}^{(j)}\f$                                       |
| `NucleationBarrierAndCriticalRadius`  | Computes \f$R^{(j)}_{\text{crit}}\f$ and \f$\Delta G^{*,(j)}\f$                            |
| `ZeldovichFactor`                     | Computes \f$Z^{(j)}\f$                                                                     |
| `KineticFactor`                       | Computes \f$\beta^{(j)}\f$                                                                 |
| `NucleationFluxMagnitude`             | Assembles the prefactor \f$Z^{(j)} \beta^{(j)} N_0^{(j)} \exp(-\Delta G^{*,(j)}/k_B T)\f$  |

The Dirac delta in the nucleation flux and the spatial discretization of the population balance equation are provided by the [finite volume](@ref finite-volume) module — `DumpInSmallestBin` for the nucleation source, and `LinearlyInterpolateToCellEdges`, `FiniteVolumeUpwindedAdvectiveFlux`, `FiniteVolumeAppendBoundaryCondition`, and `FiniteVolumeGradient` for the advection of \f$n^{(j)}\f$ in radius space. Refer to [Syntax Documentation](@ref syntax-models) for the complete catalog of options for each KWN object.

## Composition pattern

Putting these pieces together, a single-precipitate KWN model is built as follows. The example is the growth-only Al–Cu input file `tests/regression/kwn/growth-only-scaled/model.i`: a uniform grid in scaled radius coordinates is mapped to a semi-infinite physical radius via `SemiInfiniteScalingScalar`, the matrix concentration and growth rate are computed from the current size distribution, and the population balance is advanced in time by an `ImplicitUpdate` Newton solve.

@list-input:tests/regression/kwn/growth-only-scaled/model.i:Models,EquationSystems,Solvers

The growth-rate piece is responsible for setting \f$\dot{R}^{(j)}\f$ at every cell center; everything from `scaled_cell_velocity` onward is generic finite-volume advection of \f$n^{(j)}\f$ in radius space and is identical between the growth-only and full nucleation+growth cases.

To add nucleation, the same model is augmented with `NucleationBarrierAndCriticalRadius`, `ZeldovichFactor`, `KineticFactor`, `NucleationFluxMagnitude`, and `DumpInSmallestBin`, and the resulting source term is added to the flux divergence via a `ScalarLinearCombination`. The full composition is shown in `tests/regression/kwn/growth-nucleation-scaled/model.i`.

## Tests and examples

Unit tests for each KWN object live in `tests/unit/models/kwn/`, one input file per class. They exercise the forward evaluation and AD derivatives in isolation. For instance, the `PrecipitateVolumeFraction` unit test fixes a three-bin radius grid and verifies the analytic value of \f$f^{(j)}\f$:

@list-input:tests/unit/models/kwn/PrecipitateVolumeFraction.i

End-to-end regression tests live in `tests/regression/kwn/`:

- `growth-only-scaled/` — single-species growth with a tabulated chemical potential, semi-infinite scaled radius grid.
- `growth-nucleation-scaled/` — full nucleation + growth on a scaled radius grid.
- `growth-nucleation-unscaled/` — same physics on a fixed-extent radius grid (useful for comparing the effect of the semi-infinite scaling).

A worked Al–Cu example computing a time–temperature–precipitation diagram for the \f$\theta\f$ phase \f$\mathrm{Al}_2\mathrm{Cu}\f$ lives under `python/examples/kwn/`. The notebook `al-cu-ttp.ipynb` walks through generating the required CALPHAD data with [pycalphad](https://pycalphad.org) using the [Liang and Schmid-Fetzer database](https://www.sciencedirect.com/science/article/pii/S0364591615300304), then loads `model.i` and runs the resulting model on a temperature/time grid (using CUDA when available). See the example's `requirements.txt` for the additional Python dependencies needed to regenerate the thermodynamic data.

A second worked example under `python/examples/316-precipitation/` builds a two-phase KWN model for 316H stainless steel (intragranular \f$\mathrm{Cr}_{23}\mathrm{C}_6\f$ + \f$\mathrm{Fe}_2\mathrm{Mo}\f$ Laves), following the precipitation sub-model of [Hu et al., 2020](https://doi.org/10.1016/j.msea.2019.138787). It demonstrates `IdealSolutionVolumetricDrivingForce` driving the nucleation barrier for both compound and single-element precipitates, two populations sharing a common matrix solute reservoir via `CurrentConcentration`, and the rate-limited growth-rate form with a different rate-limiting species per phase.
