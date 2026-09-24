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

"""Emit the ``.i`` driver for each sweep case defined in ``make_sweep_csvs.py``.

The sweep list is imported rather than duplicated, so the CSVs and the input
files cannot drift apart.

    python make_sweep_ifiles.py
"""

from __future__ import annotations

from pathlib import Path

from make_sweep_csvs import sweep_curves

_HERE = Path(__file__).parent

_TEMPLATE = """# Cross-implementation check: rate-independent Simo-Ju + Weibull.
# Case: {curve_name}
# Parameters: Y_in = {Y_in:g},  p1 = {p1:g},  p2 = {p2:g}
#
# The reference CSV is produced by make_sweep_csvs.py from an independent
# numpy implementation of the same equations, so the composed NEML2 model
# must reproduce it to machine precision. This checks the NEML2 wiring --
# composition, chain rule, driver -- and pins behaviour across the sweep.
#
# It does NOT verify the constitutive law against a publication: a
# transcription error would appear identically on both sides. The case that
# can falsify the model is ../brandyberry_fig11a_Yin100/, whose reference is
# digitized from a published figure.
#
# Note this chain is rate-independent. Brandyberry's published curves include
# viscous relaxation (mu_visc = 20), which is why these directories are not
# named after their figures.

[Tensors]
  [times]
    type = CSVScalar
    csv_file = '{csv_name}'
    variable = 'time'
  []
  [strains]
    type = CSVSR2
    csv_file = '{csv_name}'
    variable = 'strain'
  []
  [stresses]
    type = CSVSR2
    csv_file = '{csv_name}'
    variable = 'stress'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
    save_as = 'result.pt'
  []
  [verification]
    type = Verification
    driver = 'driver'
    SR2_names = 'output.sigma'
    SR2_values = 'stresses'
    rtol = 1e-6
    atol = 1e-3
  []
[]

[Models]
  [effective_stress]
    type              = LinearIsotropicElasticity
    coefficients      = '2.5e9 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain            = 'E'
    stress            = 'sigma_tilde'
  []
  [strain_energy]
    type              = LinearIsotropicStrainEnergyDensity
    coefficients      = '2.5e9 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    decomposition     = 'NONE'
    strain            = 'E'
    active_strain_energy_density   = 'psi0'
    inactive_strain_energy_density = 'psi0_unused'
  []
  [damage_history]
    type = IrreversibleScalar
    from = 'psi0'
    to   = 'r'
  []
  [weibull]
    type = WeibullDamage
    r    = 'r'
    D    = 'D_step'
    Y_in = {Y_in:g}
    p1   = {p1:g}
    p2   = {p2:g}
  []
  [damage_monotone]
    type = IrreversibleScalar
    from = 'D_step'
    to   = 'D'
  []
  [damaged_stress]
    type             = DamagedStress
    damage           = 'D'
    effective_stress = 'sigma_tilde'
    stress           = 'sigma'
  []
  [model]
    type               = ComposedModel
    models             = 'effective_stress strain_energy damage_history
                          weibull damage_monotone damaged_stress'
    additional_outputs = 'D r psi0 sigma_tilde'
  []
[]
"""


def main() -> None:
    curves = sweep_curves()
    for name, params in curves:
        out_dir = _HERE / name
        out_dir.mkdir(parents=True, exist_ok=True)
        i_path = out_dir / f"{name}.i"
        i_path.write_text(_TEMPLATE.format(curve_name=name, csv_name=f"{name}.csv", **params))
        print(f"  wrote {i_path.relative_to(_HERE)}")
    print(f"\nGenerated {len(curves)} sweep input files under {_HERE}")


if __name__ == "__main__":
    main()
