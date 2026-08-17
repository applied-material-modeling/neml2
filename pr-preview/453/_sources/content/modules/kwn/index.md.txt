(modules-kwn)=
# KWN (precipitation kinetics)

## Overview

The KWN module is a collection of `Model` primitives for composing
Kampmann–Wagner Numerical (KWN) models of precipitation, growth, and
coarsening in a multi-component matrix. KWN models track a discretized
size distribution of precipitates; nucleation, growth, and coarsening
(Ostwald ripening) all emerge from the same population balance equation
applied to that distribution. The notation here follows
[Ury et al., 2023](https://doi.org/10.1016/j.actamat.2023.118988), and
the SFFK driving-force formulation follows
[Svoboda et al., 2004](https://doi.org/10.1016/j.msea.2004.06.018).

A complete KWN model is assembled by composing the precipitation
primitives in this module with the [](modules-finite-volume) module,
which provides the spatial discretization of the radius-space
population balance equation. The KWN primitives compute the
*physics* — volume fractions, matrix concentrations, growth rates,
nucleation barriers and prefactors — and hand the resulting cell
velocity and source term off to the generic finite-volume advection
machinery.

## Math

The size distribution of precipitate $(j)$ is discretized into
finite-volume bins on the radius axis. Subscripts $i$ index the radius
bin (with center $R^{(j)}_i$ and width
$\Delta R^{(j)}_i = R^{(j)}_{i+1/2} - R^{(j)}_{i-1/2}$); parenthetical
superscripts $(j)$ index the precipitate species. Matrix solute
concentrations are stored as mole fractions $x_k$ over the union
$\mathcal{S} = \bigcup_j \mathcal{S}_j$ of species that participate in
any precipitate reaction.

### Mass conservation

The volume fraction of precipitate $(j)$ is the third moment of its
number-density distribution,

$$
f^{(j)} = \sum_i \tfrac{4}{3}\pi \left(R_i^{(j)}\right)^3 n_i^{(j)},
$$

and the current matrix concentration of species $k$ follows from the
closure that everything not locked up in precipitates remains in
solution,

$$
x^\infty_k =
  \frac{x_{0,k} - \sum_j f^{(j)} x^{(j)}_k}
       {1 - \sum_j f^{(j)}},
$$

where $x_{0,k}$ is the initial matrix concentration of species $k$,
$x^{(j)}_k$ is the concentration of species $k$ in precipitate $(j)$
(taken as zero for $k \notin \mathcal{S}_j$), and the initial
precipitate volume fraction is assumed to be zero.

### Population balance

Each precipitate population evolves according to a 1D advection
equation in radius space with a nucleation source on the right,

$$
\frac{\partial n^{(j)}}{\partial t}
  + \frac{\partial \bigl(n^{(j)} \dot{R}^{(j)}\bigr)}{\partial R^{(j)}}
  = J^{(j)}_{\text{nuc}}.
$$

The KWN module supplies $\dot{R}^{(j)}$ at cell centers and
$J^{(j)}_{\text{nuc}}$; the [](modules-finite-volume) module
interpolates the velocity to cell edges, computes the upwinded
advective flux, applies boundary conditions, and takes the divergence
to recover $\partial n^{(j)}/\partial t$.

### Growth rate

Two growth-rate forms are provided.

The simpler **rate-limited** form solves for a single rate-limiting
species $\star \in \mathcal{S}_j$,

$$
\dot{R}^{(j)} = \frac{1}{R^{(j)}} \, D_\star \,
  \frac{x^\infty_\star - x^{*,(j)}_\star}
       {x^{(j)}_\star - x^{*,(j)}_\star},
$$

with $x^{*,(j)}_\star$ the matrix equilibrium concentration of the
rate-limiting species in equilibrium with precipitate $(j)$. The
growth rate vanishes as $x^\infty_\star \rightarrow x^{*,(j)}_\star$.

The more general **SFFK** form (Svoboda–Fischer–Fratzl–Kozeschnik)
handles multi-component diffusion in the dilute, diffusion-limited
regime,

\begin{align}
\dot{R}^{(j)} &= \frac{1}{R^{(j)}} \,
  \frac{\Delta G^{(j)}_{\text{chem}}
      + \Delta G^{(j)}_{\text{surf}}
      + \Delta G^{(j)}_{\text{el}}}
       {R_g T \, S^{(j)}}, \\
S^{(j)} &= \sum_{k \in \mathcal{S}_j}
  \frac{\bigl(\Delta x^{(j)}_k\bigr)^2}{D_k \, x^\infty_k},
\end{align}

with $\Delta x^{(j)}_k = x^{(j)}_k - x^{*,(j)}_k$ and $D_k$ the matrix
diffusivity of species $k$. The driving force is split into chemical,
surface, and elastic contributions:

\begin{align}
\Delta G^{(j)}_{\text{chem}} &=
  \sum_{k \in \mathcal{S}_j} \Delta x^{(j)}_k
  \bigl[\mu^{\text{matrix}}_k(x^\infty) - \mu^{\text{equil},(j)}_k\bigr],
  \\
\Delta G^{(j)}_{\text{surf}} &=
  -\frac{2 \gamma^{(j)} V_m^{(j)}}{R^{(j)}}, \\
\Delta G^{(j)}_{\text{el}} &=
  -3 V_m^{(j)} \varepsilon_0^{(j)} \sigma_h
  + \frac{6 V_m^{(j)} \mu K}{3K + 4\mu}
    \bigl(\varepsilon_0^{(j)}\bigr)^2.
\end{align}

Here $V_m^{(j)}$ is the molar volume of the precipitate,
$\gamma^{(j)}$ its surface energy, $\varepsilon_0^{(j)}$ the misfit
strain (coherent, isotropic spherical inclusion in an isotropic matrix
with shear modulus $\mu$ and bulk modulus $K$), and $\sigma_h$ the
matrix hydrostatic stress.

### Nucleation flux

Nucleation deposits new precipitates at the critical radius,

$$
J^{(j)}_{\text{nuc}} =
  Z^{(j)} \beta^{(j)} N_0^{(j)}
  \exp\!\left(-\frac{\Delta G^{*,(j)}}{k_B T}\right)
  \delta\!\left(R^{(j)} - R^{(j)}_{\text{crit}}\right),
$$

with the classical-nucleation-theory expressions

\begin{align}
R^{(j)}_{\text{crit}} &=
  \frac{2 \gamma^{(j)} V_m^{(j)}}{\Delta g^{(j)}_v}, \\
\Delta G^{*,(j)} &=
  \frac{16 \pi}{3} \,
  \frac{\bigl(\gamma^{(j)}\bigr)^3 \bigl(V_m^{(j)}\bigr)^2}
       {\bigl(\Delta g^{(j)}_v\bigr)^2}, \\
Z^{(j)} &=
  \frac{V_m^{(j)}}{2\pi N_a \bigl(R^{(j)}_{\text{crit}}\bigr)^2}
  \sqrt{\frac{\gamma^{(j)}}{k_B T}}, \\
\beta^{(j)} &=
  \frac{4 \pi \bigl(R^{(j)}_{\text{crit}}\bigr)^2 N_a^{4/3}}
       {\bigl(V_m^{(j)}\bigr)^{4/3}} \, \frac{1}{S^{(j)}}.
\end{align}

$\Delta g^{(j)}_v$ is the volumetric driving force for nucleation of
precipitate $(j)$ (in J/mol — the per-mole convention used throughout
[](models-NucleationBarrierAndCriticalRadius)), $N_0^{(j)}$ is the
nucleation site density, $N_a$ is Avogadro's number, $k_B$ is the
Boltzmann constant, and $S^{(j)}$ is the same projected diffusivity
sum that appears in the SFFK growth rate.

The volumetric driving force $\Delta g^{(j)}_v$ can either be supplied
as a CALPHAD-derived tabulation against composition and temperature
(the pattern used by the Al–Cu example below) or assembled from the
participating species' matrix and equilibrium concentrations with
[](models-IdealSolutionVolumetricDrivingForce), which evaluates the
Hu–Cocks ideal-solution form
$\Delta g_v = R_g T \sum_k w_k \ln(c_k / c_k^{\text{eq}})$. Setting
every weight to 1 recovers the "product of all components" convention
used for compound precipitates.

:::{note}
For most systems $R_{\text{crit}}$ is much smaller than the smallest
cell size used in a practical finite-volume discretization. The Dirac
delta at $R^{(j)}_{\text{crit}}$ is therefore approximated discretely
by `DumpInSmallestBin` from the [](modules-finite-volume) module,
which deposits the entire flux magnitude into the smallest bin.
:::

## Catalog

The KWN module exposes the following primitives, each implementing
one of the equations above. They are designed to be composed inside a
`ComposedModel` (typically driven by an `ImplicitUpdate`) rather than
used in isolation.

| Type                                                       | Role                                                                                       |
| :--------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| [](models-PrecipitateVolumeFraction)                       | Sums $\tfrac{4}{3}\pi (R_i^{(j)})^3 n_i^{(j)}$ over bins to obtain $f^{(j)}$               |
| [](models-CurrentConcentration)                            | Mass-balance closure for $x^\infty_k$ given the precipitate populations                    |
| [](models-ProjectedDiffusivitySum)                         | Computes the multi-species sum $S^{(j)}$ shared by SFFK and nucleation                     |
| [](models-ChemicalGibbsFreeEnergyDifference)               | Assembles $\Delta G^{(j)}_{\text{chem}}$ from per-species potentials                       |
| [](models-IdealSolutionVolumetricDrivingForce)             | Assembles the molar driving force $R_g T \sum_k w_k \ln(c_k / c_k^{\text{eq}})$            |
| [](models-RateLimitedPrecipitateGrowthRate)                | Single-species, equilibrium-driven growth rate $\dot{R}^{(j)}$                             |
| [](models-SFFKPrecipitationGrowthRate)                     | SFFK multi-component growth rate $\dot{R}^{(j)}$                                           |
| [](models-NucleationBarrierAndCriticalRadius)              | Computes $R^{(j)}_{\text{crit}}$ and $\Delta G^{*,(j)}$                                    |
| [](models-ZeldovichFactor)                                 | Computes the Zeldovich factor $Z^{(j)}$                                                    |
| [](models-KineticFactor)                                   | Computes the attachment frequency $\beta^{(j)}$                                            |
| [](models-NucleationFluxMagnitude)                         | Assembles $Z^{(j)} \beta^{(j)} N_0^{(j)} \exp(-\Delta G^{*,(j)} / k_B T)$                  |

## Example: growth-only Al–Cu

The input below is the growth-only Al–Cu regression case
`tests/regression/kwn/growth-only-scaled/model.i`. A uniform grid in a
scaled radius coordinate $\xi \in [0, 1]$ is mapped to a semi-infinite
physical radius $R = s\,\xi / (1 - \xi)$ via the precomputed
`true_centers`, `center_jacobian`, and `center_inverse_jacobian`
tensors. The matrix concentration and growth rate are recomputed from
the current size distribution each step, and the population balance is
advanced in time by an `ImplicitUpdate` Newton solve.

```{literalinclude} ../../../../tests/regression/kwn/growth-only-scaled/model.i
:language: ini
```

### Walkthrough

- [](models-PrecipitateVolumeFraction) (`volume_fraction`) integrates
  the *physical* number density `true_number_density` over the
  physical bin centers `true_centers` to give the precipitate volume
  fraction `vf`. The unscale step below converts the solution variable
  `number_density` (which lives on the scaled grid) into
  `true_number_density` by multiplying by the inverse Jacobian.
- [](models-CurrentConcentration) (`x_Cu`) closes the matrix
  composition with $x^\infty_{\text{Cu}} = (x_{0,\text{Cu}} - f\,
  x^{(j)}_{\text{Cu}}) / (1 - f)$, given the initial Cu mole fraction
  `x0_Cu` and the precipitate Cu fraction `xp_Cu`.
- A `ScalarLinearInterpolation` (`chemical_potential_difference`)
  evaluates the tabulated chemical-potential difference
  $\mu^{\text{matrix}}_{\text{Cu}}(x^\infty_{\text{Cu}}) -
  \mu^{\text{equil}}_{\text{Cu}}$ at the current matrix concentration.
  This is the externally supplied $\Delta G^{(j)}_{\text{chem}}$
  contribution; in a CALPHAD-coupled workflow it would come from a
  pycalphad tabulation against composition and temperature.
- [](models-ProjectedDiffusivitySum) (`diffusivity_sum`) computes
  $S^{(j)} = (\Delta x^{(j)}_{\text{Cu}})^2 /
  (D_{\text{Cu}} x^\infty_{\text{Cu}})$ with a single contributing
  species, since the only diffusing solute in this example is Cu.
- [](models-SFFKPrecipitationGrowthRate) (`growth_rate`) divides the
  chemical driving force by $R_g T S^{(j)} R^{(j)}$ to give
  $\dot{R}^{(j)}$ at each cell center. (Surface and elastic terms are
  omitted in this single-species, growth-only example; both are
  optional inputs on the SFFK model.)
- `scaled_cell_velocity`, `advection_velocity`, `advective_flux`,
  `left_bc`, `right_bc`, and `flux_divergence` are generic
  finite-volume primitives from the [](modules-finite-volume) module.
  They map the cell-center growth rate to the scaled grid, interpolate
  it to cell edges, take the upwinded advective flux, append
  zero-flux Dirichlet boundary conditions at both ends, and take the
  divergence to recover $\partial n^{(j)}/\partial t$.
- `integrate_u` adds the backward-Euler residual; `implicit_rate`
  bundles every model in the residual graph; and `model_scaled`
  drives the Newton solve through `ImplicitUpdate`. The outer
  `model` then unscales the solution and exposes the diagnostic
  outputs (`number_density`, `true_number_density`, `vf`, `x_Cu`) for
  post-processing.

To add nucleation, the same skeleton is augmented with
[](models-NucleationBarrierAndCriticalRadius),
[](models-ZeldovichFactor), [](models-KineticFactor), and
[](models-NucleationFluxMagnitude); the source term is then dumped
into the smallest bin and added to the flux divergence via a
`ScalarLinearCombination`. The full composition lives in
`tests/regression/kwn/growth-nucleation-scaled/model.i`.

## Examples

End-to-end examples live in `tests/regression/kwn/`:

- `growth-only-scaled/` — single-species growth with a tabulated
  chemical potential, semi-infinite scaled radius grid.
- `growth-nucleation-scaled/` — full nucleation + growth on a
  semi-infinite scaled radius grid.
- `growth-nucleation-unscaled/` — same physics on a fixed-extent
  radius grid (useful for isolating the effect of the semi-infinite
  scaling).

## Worked examples

End-to-end notebooks that compose this catalog into full precipitation
models and run them:

```{toctree}
:maxdepth: 1

precipitation_316h
al_cu_ttp
```

## See also

- [](modules-finite-volume) — radius-space advection, boundary
  conditions, and `DumpInSmallestBin` that the KWN catalog plugs into.
- [](tutorials-models-composition) — how `ComposedModel` wires
  primitives into a single forward operator.
- [](tutorials-models-cross-referencing) — how variable names flow
  between primitives in an input file.
- [](tutorials-models-implicit-model) — how `ImplicitUpdate` wraps a
  residual model in a Newton solve, the pattern used to advance the
  population balance in time.
- [](syntax-catalog) — the per-type option reference for every model
  listed in the catalog table above.
