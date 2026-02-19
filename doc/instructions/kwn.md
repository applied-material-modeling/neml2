# KWN model.

The new submodule is called kwn, so headers go in `include/neml2/models/kwn`, source in `src/neml2/models/kwn`, unit tests in `tests/unit/models/kwn` etc.

## Conventions

Overall follow the approach described [here](https://www.sciencedirect.com/science/article/pii/S1359645423003191).

The population balance side of the model works with $n_i$ the number of precipitates per unit volume with radius between $[R_{i-1/2}, R_{i+1/2}]$ with bin center $R_i$ and bin width $\Delta R_i$.

We will work with mole fraction for the concentrations, call them $x_i$. We can imagine this as having units of mol/mol. We can convert to weight fraction as

$$
 w_i = \frac{x_i M_i}{\sum_i x_i M_i}
$$

with $M_i$ the molar mass or vice-versa

$$
 x_i = \frac{w_i \sum_i x_i M_i}{M_i}
$$

## Input

### Constants

1. Gas constant $R$
2. Boltzmann constant $k$
3. Avagradro's number $N_A$

### General input

1. The initial concentration of the solution $x_{0,i}$
2. The surface energy of the precipitate $\gamma$ as a function of temperature.
3. The molar volume of the precipitate $V_m$.
4. The elastic constants of the matrix $K$ and $\mu$.
5. The misfit strain $\varepsilon_0$.

### CALPHAD data

1. The equilibrium concentration of the matrix $x^*$ as a function of temperature.
2. The concentration of the precipitates $x^p$ as a function of temperature. Or we can just assume its stoichiometric.
3. The chemical potential of the matrix $\mu^{matrix}$ as a function of composition and temperature.
4. The equilibrium chemical potential of each matrix-precipitate pair $\mu^{equil}$ as a function of temperature.

### Diffusion data

1. The temperature-dependent diffusivity of each species in solution $D$.

## Equations

### Mass conservation

The volume fraction of the precipitate is

$$
 f_k = \sum_j \frac{4}{3}\pi R_i^3 n_i
$$

dropping the precipitate subscript $k$ on the RHS, with units of volume per volume.

The current concentrations in solution are

$$
 x^\infty_i = \frac{x_{0,i}-\sum_k f_k x_{k}^p}{1 - \sum_k f_k}
$$

with the sums over all precipitates.

### Population balance

Each precipitate maintains its own population balance model.

$$
 \frac{d n}{d t} + \frac{d \left(n \dot{R} \right)}{dR} = J_{nuc}
$$

### Growth rate

Two options.

#### Rate-limited

$$
 \dot{R} = \frac{1}{R} D_{\eta \eta} \frac{x^\infty_\eta - x^*_\eta}{x^p_\eta - x^*_\eta}
$$

for the limiting species $\eta$ and with $x^*$ the equilibrium concentration of the matrix, which we can tabulate as a function of temperature.

As a sanity check, as the matrix concentration approaches the equilibrium concentration this rate clearly goes to zero. The units are also correct: the growth rate should have units of $L/T$ and this has units of $\frac{1}{L} \frac{L^2}{T} = \frac{L}{T}$.

#### SFFK

Based on [Svoboda 2004](https://www.sciencedirect.com/science/article/pii/S0921509304008202) in the dilute limit where precipitates do not interact with each other, just interact pairwise through the matrix chemistry. In this case we can again drop the precipitate subscript as they are all independent. We will also drop the interface mobility and assume we are diffusion limited.

$$
 \dot{R} = \frac{1}{R} \frac{\Delta G_{chem} - \Delta G_{surf} - \Delta G_{el}}{R T \sum_i \frac{\Delta x_i^2}{D_i x^\infty_i}}
$$

with

$$
 \Delta x_i = x^p_i - x^\star_i
$$

and

$$
 \Delta G_{chem} = \sum_i \Delta x_i \left[\mu^{matrix}(x^\infty) - \mu^{equil}\right]
$$

$$
 \Delta G_{surf} = -\frac{2\gamma V_m}{R}
$$

The elastic term is debated, but we can start with

$$
 \Delta G_{el} = -3V_m \varepsilon_0 \sigma_h + \frac{V_m 6 \mu K}{3K+4\mu}\varepsilon_0^2
$$

for a simple, coherent, isotropic inclusion in an isotropic matrix with a given misfit strain $\varepsilon_0$.

Checking dimensional consistency of the Gibbs energy differences:

1. $\Delta G_{chem} = E/mol$
2. $\Delta G_{surf} = L^3 E / (mol \cdot L \cdot L^2) = E/mol$
3. First term: $\Delta G_{el} = L^3 / mol \cdot F / L^2 = E/mol$
4. Second term: same units

and then the overall growth rate: $1/L \cdot E / mol \cdot (L^2/T)/(E/mol) = L/T$. So good.

### Nucleation flux

All precipitates nucleate at the critical radius:

$$
 J_{nuc} = Z \beta N_0 \exp\left(\frac{\Delta G}{kT}\right) \delta(R - R_{crit})
$$

with

$$
 R_{crit} = \frac{2\gamma}{\Delta g_v}
$$

$$
 \Delta G = \frac{16\pi \gamma^3}{3 \Delta g_v^2}
$$

and, for spherical particles,

$$
 Z = \frac{V_m}{2\pi R_{crit}^2 N_a} \sqrt{\frac{\gamma}{kT}}
$$

with

$$
 \beta = \frac{4\pi R^2 N_a^{4/3}}{V_m^{4/3}} \frac{1}{\sum_i \frac{\Delta x_i^2}{D_i x^\infty_i}}
$$

evaluated at $R_{crit}$. Note the common diffusion term from above.

The volumetric driving force is just

$$
 \Delta g = \frac{\Delta G_{chem} + \Delta G_{el}}{V_m}
$$

Unit check:

1. $\Delta g \sim E / L^3$
2. $\Delta G \sim \frac{E^3 L^6}{L^6 E^2} = E$
3. So $\exp\left(\frac{\Delta G}{kT}\right)$ is correctly unitless.
4. $N_0 \sim 1 / L^3$
5. $Z \sim \frac{L^3 mol}{mol \cdot L^2} \sqrt{\frac{E}{L^2 E}}$ which is unitless
6. $\beta \sim \frac{L^2 mol^{4/3}}{ mol^{4/3} L^4} \frac{L^2}{T} = 1/T$

So overall the expression has units of $1/(L^3 T)$ which is right.
