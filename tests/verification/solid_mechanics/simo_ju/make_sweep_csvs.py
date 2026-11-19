# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Parameter-sweep cross-checks for the rate-independent Simo-Ju + Weibull chain.

What these are
--------------
Each CSV here is produced by the independent numpy implementation below and
compared against the composed NEML2 model to machine precision. That makes
them **cross-implementation checks**: they catch composition, chain-rule and
driver bugs in the NEML2 wiring, and they pin behaviour across a wide span of
(Y_in, p1, p2).

What these are NOT
------------------
They are **not** verification against a publication. Reference and model
implement the same equations, so a transcription error in the constitutive law
would appear identically on both sides and every case here would still pass.
For a reference that can actually falsify the model, see
``make_digitized_csvs.py`` and ``brandyberry_fig11a_Yin100/``, whose values are
digitized from a published figure.

Sweep ranges are chosen to bracket the behaviour Brandyberry et al. (2022)
explore -- a Y_in sweep, a p1 sweep and a p2 sweep about the same nominal point
-- but the directories are deliberately not named after their figures, because
the published curves are the *viscous* model (mu_visc = 20) whereas the chain
exercised here is rate-independent.

CSV format (13 columns):
    time, strain_xx, strain_yy, strain_zz, strain_yz, strain_xz, strain_xy,
          stress_xx, stress_yy, stress_zz, stress_yz, stress_xz, stress_xy

Uniaxial *stress* state: lateral strains are -nu*eps_xx, so the transverse
effective stresses vanish and only sigma_xx is nonzero.

Re-run if the reference formulas ever change:
    python make_sweep_csvs.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

# --------------------------------------------------------------------------
# Material + loading configuration
# --------------------------------------------------------------------------

E = 2.5e9  # Pa
NU = 0.3
STRAIN_MAX = 1.0e-3  # 1 permille
N_STEPS = 200
T_TOTAL = 1.0  # s -- constant strain rate of 1e-3 1/s


def simulate_uniaxial(
    strain: np.ndarray, E: float, Y_in: float, p1: float, p2: float
) -> dict[str, np.ndarray]:
    """Rate-independent Simo-Ju damage with a three-parameter Weibull law.

    Deliberately self-contained: this file must run from a fresh clone, so it
    carries its own reference rather than importing one from elsewhere in the
    tree.

        psi0 = 1/2 E eps^2                      (uniaxial stress)
        r    = max-history of psi0              (IrreversibleScalar)
        D    = 1 - exp[-(<r - Y_in>/(p1 Y_in))^p2]
        sig  = (1 - D) E eps
    """
    psi0 = 0.5 * E * strain**2
    r = np.maximum.accumulate(psi0)
    arg = np.maximum(r - Y_in, 0.0) / (p1 * Y_in)
    D = np.maximum.accumulate(1.0 - np.exp(-(arg**p2)))
    return {"psi0": psi0, "damage": D, "stress": (1.0 - D) * E * strain}


# --------------------------------------------------------------------------
# Sweep definitions -- single source of truth, also imported by
# make_sweep_ifiles.py so the two generators cannot drift apart.
# --------------------------------------------------------------------------


def sweep_curves() -> list[tuple[str, dict[str, float]]]:
    """(directory name, parameters) for every sweep case."""
    curves: list[tuple[str, dict[str, float]]] = []

    def slug(value: float) -> str:
        return str(int(value)) if value == int(value) else str(value).replace(".", "p")

    for p1 in (0.01, 0.1, 1.0, 10.0, 100.0):  # p1 sweep
        curves.append((f"sweep_p1_{slug(p1)}", {"Y_in": 300.0, "p1": p1, "p2": 1.0}))
    for Y_in in (100.0, 200.0, 300.0, 400.0, 500.0):  # Y_in sweep
        curves.append((f"sweep_Yin_{slug(Y_in)}", {"Y_in": Y_in, "p1": 1.0, "p2": 1.0}))
    for p2 in (0.01, 0.1, 1.0, 10.0, 100.0):  # p2 sweep
        curves.append((f"sweep_p2_{slug(p2)}", {"Y_in": 300.0, "p1": 1.0, "p2": p2}))
    return curves


HEADER = [
    "time",
    "strain_xx",
    "strain_yy",
    "strain_zz",
    "strain_yz",
    "strain_xz",
    "strain_xy",
    "stress_xx",
    "stress_yy",
    "stress_zz",
    "stress_yz",
    "stress_xz",
    "stress_xy",
]


def write_csv(out_path: Path, strain: np.ndarray, stress: np.ndarray) -> None:
    """Write one 13-column CSV, prefixed with the virgin t = 0 state."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dt = T_TOTAL / N_STEPS
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerow(["0"] + ["0"] * 12)
        for k, (eps, sig) in enumerate(zip(strain, stress, strict=True), start=1):
            lat = -NU * eps
            row = [k * dt, eps, lat, lat, 0.0, 0.0, 0.0, sig, 0.0, 0.0, 0.0, 0.0, 0.0]
            w.writerow([f"{x:.18e}" for x in row])


def main() -> None:
    strain = np.linspace(STRAIN_MAX / N_STEPS, STRAIN_MAX, N_STEPS)
    print(
        f"Simo-Ju sweep CSV generator "
        f"(E={E:.2e} Pa, strain 0 to {STRAIN_MAX} over {T_TOTAL} s in {N_STEPS} steps)"
    )
    curves = sweep_curves()
    for name, params in curves:
        result = simulate_uniaxial(strain, E=E, **params)
        out_path = _HERE / name / f"{name}.csv"
        write_csv(out_path, strain, result["stress"])
        print(
            f"  wrote {out_path.relative_to(_HERE)}  "
            f"(Y_in={params['Y_in']:g}, p1={params['p1']:g}, p2={params['p2']:g}, "
            f"peak_sigma={result['stress'].max() * 1e-6:.3f} MPa)"
        )
    print(f"\nGenerated {len(curves)} sweep CSVs under {_HERE}")


if __name__ == "__main__":
    main()
