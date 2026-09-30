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

"""Convert curves digitized from published figures into NEML2 verification CSVs.

Unlike ``make_sweep_csvs.py`` -- whose reference values are produced by an
independent implementation of the *same* equations, and therefore cannot
detect a modelling error -- the references here are read off a published
plot. They are the only cases in this directory that can falsify the model.

Provenance
----------
Both cases are from Brandyberry, D. R., Zhang, X. & Geubelle, P. H. (2022),
*Comput. Methods Appl. Mech. Engrg.* **399**, 115388. Each raw digitizer
export (x = axial strain [-], y = axial stress [MPa]) is committed alongside
its converted CSV so the conversion can be re-checked.

* ``brandyberry_fig11a_Yin100`` -- Fig. 11(a), the Y_in = 100 curve
  (Y_in = 100, p1 = 1, p2 = 1).
* ``brandyberry_fig11b_p2_10`` -- Fig. 11(b), the p2 = 10 curve
  (Y_in = 300, p1 = 1, p2 = 10).

The second case is not redundant. At p1 = 1 and p2 = 1 the damage law
degenerates -- ``(arg/(p1*Y_in))^p2`` loses both parameters -- so the Fig.
11(a) curve is *algebraically incapable* of detecting an error in how p1 or
p2 enter. Coding ``arg^p2`` as ``arg*p2``, or ``p1*Y_in`` as ``p1+Y_in-1``,
changes it by exactly zero. The p2 = 10 curve is off that degenerate point
and does discriminate.

Loading schedule
----------------
The published curves are the *viscous* model (mu_visc = 20 1/s), not the
rate-independent one. Fitting the single unknown time scale to the
digitized curve gives mu*T = 19.92, i.e. T = 1 s to reach eps = 1e-3 --
a strain rate of 1e-3 1/s. Under a constant strain rate, each digitized
point's time follows from its strain, t = eps / eps_rate.

Raw digitizer output is not sorted (points near the peak come out of
order) and its first sample sits slightly left of the origin; both are
cleaned up here rather than by hand-editing the raw file.

CSV format (13 columns, matching the rest of this directory):
    time, strain_xx, strain_yy, strain_zz, strain_yz, strain_xz, strain_xy,
          stress_xx, stress_yy, stress_zz, stress_yz, stress_xz, stress_xy

Uniaxial *stress* state: lateral strains are -nu*eps_xx so that the
transverse effective stresses vanish and only sigma_xx is nonzero.

Re-run after re-digitizing:
    python make_digitized_csvs.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

NU = 0.3
STRAIN_RATE = 1.0e-3  # 1/s -- eps = 1e-3 reached at t = 1 s

# scenario -> raw digitized file (x = strain, y = stress in MPa)
CASES = {
    "brandyberry_fig11a_Yin100": "raw_digitized_fig11a_Yin100.csv",
    "brandyberry_fig11b_p2_10": "raw_digitized_fig11b_p2_10.csv",
}

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


def load_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column digitizer export, sort it, and drop origin noise."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    raw = raw[np.argsort(raw[:, 0])]
    # Digitizers routinely emit a first sample a hair left of the axis; it
    # carries no information and would give the driver a negative strain.
    raw = raw[raw[:, 0] > 1.0e-8]
    return raw[:, 0], raw[:, 1]


def write_csv(out_path: Path, strain: np.ndarray, stress_mpa: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerow(["0"] + ["0"] * 12)  # virgin state at t = 0
        for eps, sig in zip(strain, stress_mpa, strict=True):
            lat = -NU * eps
            row = [
                eps / STRAIN_RATE,
                eps,
                lat,
                lat,
                0.0,
                0.0,
                0.0,
                sig * 1.0e6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            w.writerow([f"{x:.18e}" for x in row])


def main() -> None:
    for scenario, raw_name in CASES.items():
        strain, stress = load_raw(_HERE / scenario / raw_name)
        out = _HERE / scenario / f"{scenario}.csv"
        write_csv(out, strain, stress)
        print(
            f"  wrote {out.relative_to(_HERE)}  "
            f"({len(strain)} digitized points, "
            f"eps {strain.min():.3e}..{strain.max():.3e}, "
            f"peak {stress.max():.4f} MPa)"
        )


if __name__ == "__main__":
    main()
