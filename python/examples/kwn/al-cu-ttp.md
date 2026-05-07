---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: neml2
  language: python
  name: python3
---

# Al-Cu alloy TTP diagram
This example simulates TTP diagrams for an Al-Cu alloy, looking at the eventual nucleation and growth of the $\theta$ phase $\mathrm{Al}_2 \mathrm{Cu}$ precipitate from the Al-Cu matrix.  The actual precipitation of this phase is more complicated than its made out to be in this model, going through several metastable phases prior to reaching the stable $\mathrm{Al}_2 \mathrm{Cu}$.  This model neglects these complications in favor of a simple model.

The notebook is split into two parts.

1. The first part calculates the required thermodynamic data required to parameterize the KWN model in neml2 using [pycalphad](pycalphad.org) and the thermodynamic database provided by Liang and Schmid-Fetzer in [their 2015 paper](https://www.sciencedirect.com/science/article/pii/S0364591615300304).  Make sure you install the dependencies from the supplemental `requirements.txt` file in this directory needed to run these calculations.  This information is then included in the NEML2 material model.  This directory also contains the final NEML2 material model, with the thermodynamic parameters.
2. The second part of the notebook loads this model and evaluates the TTP diagram (using CUDA, if available) for this system.  Al-Cu is very well studied and know to be age hardenable at temperatures slightly above room temperature.  Without fitting, the model is able to reasonably predict the knee of the TTP diagram at approximately the correct temperature.

Note running these examples requires some additional python packages, listed in the `requirements.txt` in this folder.

+++

### Define alloy composition and conditions

Set the actual Cu atomic fraction in our alloy along with the conditions used to run the reference phase diagram calculations.

```{code-cell} ipython3
initial_cu_fraction = 0.01738 # Initial Cu fraction in the alloy, this is 4 wt%
temperatures = (300,800,20) # Temperature range in pycalphad format (min, max, step)
p = 101325 # Pressure in Pa
N = 1 # Number of moles
cu_range_consider = (0,0.02,0.00025) # Range of Cu concentrations in the matrix for the model to consider (min, max, step)
```

+++ {"vscode": {"languageId": "plaintext"}}

## Part 1: Generate thermodynamic data

```{code-cell} ipython3
import numpy as np
from pycalphad import Database, equilibrium, binplot, variables as v
import matplotlib.pyplot as plt
```

### Phase diagram
As a preliminary step, load the thermodynamic database and generate a phase diagram.  This serves as basic validation of the database (for example, compare to a reference phase diagram on [wikipedia](en.wikipedia.org/wiki/Aluminium–copper_alloys)).

```{code-cell} ipython3
# Load database
dbf = Database('AlCu-Liang-Schmid-Fetzer.tdb')
# Elements we care about
comps = ["CU", "AL"]
```

```{code-cell} ipython3
# Generate and plot the phase diagram
phases = ["LIQUID", "FCC", "AL2CU"]
binplot(
    dbf, comps, phases, {v.X("CU"): (0, 0.05, 0.005), v.T: (300, 1000, 10), v.P: 101325, v.N: 1}
)
plt.show()
```

### Matrix-Al2Cu equilibrium

The first set of parameters we need relates to the equilibrium between the Al-Cu matrix and the Al2Cu precipitates.  Setup the equilibrium calculation as a function of temperature and extract:
1. The equilibrium fraction of Cu in the matrix.
2. The equilibrium fraction of Cu in the precipitate.
3. The chemical potential at equilibrium, as a function of temperature.
4. Post-process the concentration difference between the Cu concentration in the precipitate and the matrix, as a function of temperature.

```{code-cell} ipython3
phases = ["FCC", "AL2CU"]
eq = equilibrium(dbf, comps, phases, {v.X("CU"): initial_cu_fraction, v.T: temperatures, v.P: p, v.N: N})
```

```{code-cell} ipython3
# Extract (and plot) the equilibrium fraction of Cu in the matrix as a function of temperature
temperatures = eq.coords["T"].values
eq_X_Cu_matrix = eq["X"][0,0,:,0,0,1].values
plt.plot(temperatures, eq_X_Cu_matrix)
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium Cu fraction in matrix")
plt.title("Equilibrium Cu fraction in matrix vs Temperature")
plt.show()
```

```{code-cell} ipython3
# Extract (and plot) the equilibrium fraction of Cu in the precipitate as a function of temperature
eq_X_Cu_precipitate = eq["X"][0,0,:,0,1,1].values
plt.plot(temperatures, eq_X_Cu_precipitate)
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium Cu fraction in precipitate")
plt.title("Equilibrium Cu fraction in precipitate vs Temperature")
plt.show()
```

```{code-cell} ipython3
# Extract and plot the checmical potential at equilibrium as a function of temperature
eq_chemical_potential = eq["MU"][0,0,:,0,1].values
plt.plot(temperatures, eq_chemical_potential)
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium chemical potential of Cu in precipitate (J/mol)")
plt.title("Equilibrium chemical potential of Cu in precipitate vs Temperature")
plt.show()
```

```{code-cell} ipython3
# The actual data that goes into the NEML2 kwn model is the difference in equilibrium concentration between the precipitate and the matrix
eq_concentration_difference = eq_X_Cu_precipitate - eq_X_Cu_matrix
plt.plot(temperatures, eq_concentration_difference)
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium Cu concentration difference")
plt.title("Concentration difference vs temperature")
plt.show()
```

### Matrix chemical potential

Now we need to calculate the chemical potential of the matrix as a function of temperature and Cu concentration.  The difference between the matrix and precipitate equilibrium potential is the chemical driving force for nucleation and growth.

```{code-cell} ipython3
phases = ["FCC"]
eq_matrix = equilibrium(
    dbf, comps, phases, {v.X("CU"): cu_range_consider, v.T: temperatures, v.P: p, v.N: N}
)
```

```{code-cell} ipython3
# Extract the chemical potential as a function of temperature and Cu concentration in the matrix
matrix_temperatures = eq_matrix.coords["T"].values
matrix_cu_concentrations = eq_matrix.coords["X_CU"].values
matrix_chemical_potential = eq_matrix["MU"][0,0,:,:,1].values
```

```{code-cell} ipython3
# Now calculate the driving force concentration_difference * (mu_matrix - mu_eq)
chemical_driving_force = (matrix_chemical_potential - eq_chemical_potential[:, np.newaxis]) * eq_concentration_difference[:, np.newaxis]
plt.plot(matrix_cu_concentrations, chemical_driving_force.T[:,::5])
plt.legend([f"T={T}K" for T in temperatures[::5]])
plt.xlabel("Cu Concentration")
plt.ylabel("Chemical Driving Force")
plt.show()
```

### Print the data we need to copy into the NEML2 model
In addition to what we print out below we need:
1. The molar volume of the precipitate -- calculated stochemetrically as 1.55e12 microns^3/mol
2. The surface energy of the preciptiate -- taken to be around 0.1 J/m^2 based on a glance through the literature.
3. The diffusivity of Cu in Al -- $0.150 \exp\left(\frac{-30200}{RT}\right)$ again based on a quick glance in the literature

```{code-cell} ipython3
print(f"Number of temperature points: {len(temperatures)}")

print("Temperature for interpolation:")
print(" ".join(map(str, temperatures)))
print("Precipitate concentration at equilibrium:")
print(" ".join(map(str, eq_X_Cu_precipitate)))
print("Concentration difference between precipitate and matrix at equilibrium:")
print(" ".join(map(str, eq_concentration_difference)))
```

```{code-cell} ipython3
print("Number of Cu concentration points for interpolation: ", len(matrix_cu_concentrations))

print("Cu concentration for interpolation:")
print(" ".join(map(str, matrix_cu_concentrations)))

print("Chemical potential difference for printing: ")
for row in chemical_driving_force:
    print(" ".join(map(str, row)))
```

## Part 2: Calculating a TTP diagram

Now let's use the model to calculate a TTP diagram for the material.  This requires simulating precpitiation over a range of temperatures and times and plotting contours at constant volume fractions of the precipitate.

```{code-cell} ipython3
import torch
import neml2
import tqdm
```

```{code-cell} ipython3
torch.set_default_dtype(torch.double)
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
torch.set_default_device(device)
```

### Define the conditions to use for calculating the TTP diagram
The TTP diagram is basically a contour plot of the volume fraction of the $\theta$ phase at different conditions of temperature and time.  To calculate the diagram then we need to batch out a large number of KWN simulations of precipitation.

The following cell defines the parameter grid: we chunk out the temperature range of interest and for each temperature run a KWN simulation out the requested time.  The cell also defines the initial conditions for the particle density.  Then we load the model from the `model.i` file.

```{code-cell} ipython3
# Parameters for loading
nsteps = 500
ntemps = 50
nbins = 300
times = torch.logspace(-4,2,nsteps)
temperatures = torch.linspace(300,450,ntemps)
# Start with a very low initial particle density to avoid numerical issues, the model should be insensitive to this as long as it is low enough
ic = torch.full((nbins,), 1e-12)
```

```{code-cell} ipython3
# Load the model
nmodel = neml2.load_model("model.i", "model")
nmodel.to(device=device)
```

### Run the simulations
In theory we could run these all at once in a large batch.  However, most systems run out of memory, so instead we'll divide the temperature conditions into smaller temperature chunks, controlled by `temp_chunk`.  Run the batch of temperatures through the full time history for each chunk.

```{code-cell} ipython3
# Temperature chunk size
temp_chunk = 5

# Integrate the model in a loop
# Shape should be (ntemps, nbins)
N = []
vf = []
for j in tqdm.trange(0, ntemps, temp_chunk):
    N.append([])
    vf.append([])
    T = neml2.Scalar(temperatures[j:j+temp_chunk],0)
    for i in range(1, len(times)):
        t_old = neml2.Scalar(times[i-1])
        t_new = neml2.Scalar(times[i])
        if i == 1:
            N_old = neml2.Scalar(ic,0,1)
        else:
            N_old = N[-1][-1]
        updated_state = nmodel.value({"T": T, "t": t_new, "t~1": t_old, "number_density~1": N_old})
        N[-1].append(updated_state["number_density"])
        vf[-1].append(updated_state["vf"])
```

### Consolidate the data and plot

Piece together the subbatches of data and plot the TTP diagram.  In actuality the knee should be about 100 K hotter than what is simulated here, but we didn't calibrate any properties and used a thermodynamic database that might not be optimized for this application

```{code-cell} ipython3
N_full = torch.stack([torch.stack(list(map(lambda x: x.torch(), n))) for n in N]).swapaxes(1,2).flatten(0,1)
vf_full = torch.stack([torch.stack(list(map(lambda x: x.torch(), v))) for v in vf]).swapaxes(1,2).flatten(0,1)
```

```{code-cell} ipython3
mesh_T, mesh_t = torch.meshgrid(temperatures, times[1:], indexing = 'ij')
CS = plt.contour(mesh_t, mesh_T, vf_full, levels = [0.01,0.02,0.03,0.04,0.05], colors = "k")
plt.clabel(CS, inline=1, fontsize=10)
plt.gca().set_xscale("log")
plt.xlim([1e-1,5e1])
plt.xlabel("Time (hr)")
plt.ylabel("Temperature (K)")
plt.title("$\mathrm{AL}_2\mathrm{Cu}$ TTP diagram")
```

### Plot some of the number density distributions

This plot shows the number density distribution from the KWN simulations at different temperatures, for the final time step.  It's hard to follow the colors, but you can see the trend indicated by the TTP diagram: you get the highest volume fraction around 360 K, diminishing at temperatures above and below this critical point.

```{code-cell} ipython3
edges = torch.logspace(-2,1.25,nbins+1)
radii = 0.5 * (edges[:-1] + edges[1:])
plt.semilogx(radii, N_full[::5,-1].T, label = [f"T={T:.0f}K" for T in temperatures[::5]])
plt.xlabel("Particle radius (mm)")
plt.ylabel("Number density (#/mm^3)")
plt.legend()
```
