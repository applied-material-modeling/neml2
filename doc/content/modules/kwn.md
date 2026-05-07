# KWN Precipitation {#kwn}

[TOC]

The KWN physics module is a collection of objects for composing Kampmann–Wagner Numerical (KWN) models of precipitation, growth, and coarsening in a multi-component matrix. KWN models track a discretized size distribution of precipitates, allowing nucleation, growth, and coarsening (Ostwald ripening) to emerge from the same population balance equation. The conventions and notation followed here are those of [Ury et al., 2023](https://doi.org/10.1016/j.actamat.2023.118988), and the SFFK driving force formulation follows [Svoboda et al., 2004](https://doi.org/10.1016/j.msea.2004.06.018).

A complete KWN model in NEML2 is composed by combining the precipitation building blocks below with the [finite volume](@ref finite-volume) module, which provides the spatial discretization of the population balance equation for the number density distribution of the precipitates as a function of their radius.

## Theory

The state of the system is the precipitate number density per unit volume, \f$n_i\f$, in each radius bin \f$\left[R_{i-1/2}, R_{i+1/2}\right]\f$ with center \f$R_i\f$. Concentrations are stored as mole fractions \f$x_i\f$. The matrix is treated as a homogeneous reservoir whose composition is determined by mass conservation against the precipitates.

### Mass conservation

The volume fraction of a given precipitate population is obtained by integrating the size distribution,

\f[
  f = \sum_i \frac{4}{3}\pi R_i^3 n_i,
\f]

and the current matrix concentration of species \f$i\f$ follows from the closure that everything not locked up in precipitates must be in solution,

\f[
  x^\infty_i = \frac{x_{0,i} - \sum_k f_k x^p_{k,i}}{1 - \sum_k f_k},
\f]

where \f$x_{0,i}\f$ is the initial concentration, \f$f_k\f$ is the volume fraction of the \f$k\f$-th precipitate, and \f$x^p_{k,i}\f$ is the concentration of species \f$i\f$ in that precipitate.

### Population balance

Each precipitate population evolves according to

\f[
  \frac{\partial n}{\partial t} + \frac{\partial \left(n \dot{R}\right)}{\partial R} = J_{\text{nuc}},
\f]

a 1D advection equation in radius space with a nucleation source on the right-hand side. Spatial discretization is provided by the [finite volume](@ref finite-volume) module: cells are defined on \f$R\f$, the growth rate \f$\dot{R}\f$ becomes the advection velocity at cell edges, and the nucleation source is dumped into the smallest bin near the critical radius via `DumpInSmallestBin`.

### Growth rate

Two growth-rate models are provided.

The simpler **rate-limited** form solves for a single rate-limiting species \f$\eta\f$,

\f[
  \dot{R} = \frac{1}{R} D_\eta \frac{x^\infty_\eta - x^*_\eta}{x^p_\eta - x^*_\eta},
\f]

with \f$x^*_\eta\f$ the matrix equilibrium concentration. The growth rate vanishes correctly as \f$x^\infty_\eta \rightarrow x^*_\eta\f$.

The more general **SFFK** form (Svoboda–Fischer–Fratzl–Kozeschnik) handles multi-component diffusion in the dilute, diffusion-limited regime,

\f[
  \dot{R} = \frac{1}{R}\, \frac{\Delta G_{\text{chem}} + \Delta G_{\text{surf}} + \Delta G_{\text{el}}}{R_g T \, S},
  \qquad
  S = \sum_i \frac{(\Delta x_i)^2}{D_i x^\infty_i},
\f]

with \f$\Delta x_i = x^p_i - x^*_i\f$. The driving force is split into chemical, surface, and elastic contributions:

\f[
  \Delta G_{\text{chem}} = \sum_i \Delta x_i \left[\mu^{\text{matrix}}_i(x^\infty) - \mu^{\text{equil}}_i\right], \quad
  \Delta G_{\text{surf}} = -\frac{2 \gamma V_m}{R}, \quad
  \Delta G_{\text{el}} = -3 V_m \varepsilon_0 \sigma_h + \frac{6 V_m \mu K}{3K + 4\mu}\varepsilon_0^2.
\f]

The \f$\Delta G_{\text{el}}\f$ term is the standard expression for a coherent, isotropic spherical inclusion in an isotropic matrix with misfit strain \f$\varepsilon_0\f$.

### Nucleation flux

All nucleation occurs at the critical radius,

\f[
  J_{\text{nuc}} = Z \beta N_0 \exp\left(-\frac{\Delta G^*}{kT}\right) \delta(R - R_{\text{crit}}),
\f]

with the classical-nucleation-theory expressions

\f[
  R_{\text{crit}} = \frac{2 \gamma V_m}{\Delta g_v}, \qquad
  \Delta G^* = \frac{16 \pi}{3}\, \frac{\gamma^3 V_m^2}{(\Delta g_v)^2},
\f]

\f[
  Z = \frac{V_m}{2\pi N_a R_{\text{crit}}^2}\sqrt{\frac{\gamma}{kT}}, \qquad
  \beta = \frac{4 \pi R_{\text{crit}}^2 N_a^{4/3}}{V_m^{4/3}}\, \frac{1}{S}.
\f]

\f$\gamma\f$ is the precipitate surface energy, \f$V_m\f$ its molar volume, \f$N_0\f$ the nucleation site density, \f$N_a\f$ Avogadro's number, \f$k\f$ the Boltzmann constant, and \f$S\f$ the same projected diffusivity sum that appears in the SFFK growth rate. The Dirac delta is approximated discretely by `DumpInSmallestBin`, which deposits the flux magnitude into the bin containing the critical radius.

## Building blocks

The KWN module exposes the following objects, each implementing one of the equations above. They are designed to be composed together (typically inside a `ComposedModel` driven by `ImplicitUpdate`) rather than used in isolation.

| Object                                | Role                                                                       |
| :------------------------------------ | :------------------------------------------------------------------------- |
| `PrecipitateVolumeFraction`           | Sums \f$\tfrac{4}{3}\pi R^3 n\f$ over bins to obtain \f$f\f$               |
| `CurrentConcentration`                | Mass-balance closure for \f$x^\infty\f$ given the precipitate populations  |
| `ProjectedDiffusivitySum`             | Computes the multi-species sum \f$S\f$ shared by SFFK and nucleation       |
| `ChemicalGibbsFreeEnergyDifference`   | Assembles \f$\Delta G_{\text{chem}}\f$ from per-species potentials         |
| `RateLimitedPrecipitateGrowthRate`    | Single-species, equilibrium-driven growth rate \f$\dot{R}\f$               |
| `SFFKGPrecipitationGrowthRate`        | SFFK multi-component growth rate \f$\dot{R}\f$                             |
| `NucleationBarrierandCriticalRadius`  | Computes \f$R_{\text{crit}}\f$ and \f$\Delta G^*\f$                        |
| `ZeldovichFactor`                     | Computes \f$Z\f$                                                           |
| `KineticFactor`                       | Computes \f$\beta\f$                                                       |
| `NucleationFluxMagnitude`             | Assembles the prefactor \f$Z \beta N_0 \exp(-\Delta G^*/kT)\f$             |

The Dirac delta in the nucleation flux and the spatial discretization of the population balance equation are provided by the [finite volume](@ref finite-volume) module — `DumpInSmallestBin` for the nucleation source, and `LinearlyInterpolateToCellEdges`, `FiniteVolumeUpwindedAdvectiveFlux`, `FiniteVolumeAppendBoundaryCondition`, and `FiniteVolumeGradient` for the advection of \f$n\f$ in radius space. Refer to [Syntax Documentation](@ref syntax-models) for the complete catalog of options for each KWN object.

## Composition pattern

Putting these pieces together, a single-precipitate KWN model is built as follows. The example is the growth-only Al–Cu input file `tests/regression/kwn/growth-only-scaled/model.i`: a uniform grid in scaled radius coordinates is mapped to a semi-infinite physical radius via `SemiInfiniteScalingScalar`, the matrix concentration and growth rate are computed from the current size distribution, and the population balance is advanced in time by an `ImplicitUpdate` Newton solve.

@list-input:tests/regression/kwn/growth-only-scaled/model.i:Models,EquationSystems,Solvers

The growth-rate piece is responsible for setting \f$\dot{R}\f$ at every cell center; everything from `scaled_cell_velocity` onward is generic finite-volume advection of \f$n\f$ in radius space and is identical between the growth-only and full nucleation+growth cases.

To add nucleation, the same model is augmented with `NucleationBarrierandCriticalRadius`, `ZeldovichFactor`, `KineticFactor`, `NucleationFluxMagnitude`, and `DumpInSmallestBin`, and the resulting source term is added to the flux divergence via a `ScalarLinearCombination`. The full composition is shown in `tests/regression/kwn/growth-nucleation-scaled/model.i`.

## Tests and examples

Unit tests for each KWN object live in `tests/unit/models/kwn/`, one input file per class. They exercise the forward evaluation and AD derivatives in isolation. For instance, the `PrecipitateVolumeFraction` unit test fixes a three-bin radius grid and verifies the analytic value of \f$f\f$:

@list-input:tests/unit/models/kwn/PrecipitateVolumeFraction.i

End-to-end regression tests live in `tests/regression/kwn/`:

- `growth-only-scaled/` — single-species growth with a tabulated chemical potential, semi-infinite scaled radius grid.
- `growth-nucleation-scaled/` — full nucleation + growth on a scaled radius grid.
- `growth-nucleation-unscaled/` — same physics on a fixed-extent radius grid (useful for comparing the effect of the semi-infinite scaling).

A worked Al–Cu example computing a time–temperature–precipitation diagram for the \f$\theta\f$ phase \f$\mathrm{Al}_2\mathrm{Cu}\f$ lives under `python/examples/kwn/`. The notebook `al-cu-ttp.ipynb` walks through generating the required CALPHAD data with [pycalphad](https://pycalphad.org) using the [Liang and Schmid-Fetzer database](https://www.sciencedirect.com/science/article/pii/S0364591615300304), then loads `model.i` and runs the resulting model on a temperature/time grid (using CUDA when available). See the example's `requirements.txt` for the additional Python dependencies needed to regenerate the thermodynamic data.
