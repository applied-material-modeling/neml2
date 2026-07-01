"""Generate v3-format verification CSVs from the digitized PCM 2001 Fig. 1 data.

Input:  ~/projects/NEML2_V3/tutorial/mazars/modified_ds.csv
        (digitized sigma_xx vs eps_xx points from PCM 2001 Fig. 1;
         63 compression points + 10 tension points)

Output:
  pcm_2001_handbook_compression/pcm_2001_handbook_compression.csv
  pcm_2001_handbook_tension/pcm_2001_handbook_tension.csv

Both CSVs follow v3's standard verification format:
  time, strain_xx, strain_yy, strain_zz, strain_yz, strain_xz, strain_xy,
        stress_xx, stress_yy, stress_zz, stress_yz, stress_xz, stress_xy

Loading is uniaxial-stress (Poisson laterals on strain), so:
  - strain_yy = strain_zz = -nu * strain_xx  (with nu = 0.2)
  - all strain shears  = 0
  - stress_yy = stress_zz = 0  (exact, for matched elastic Poisson)
  - all stress shears  = 0

Each scenario starts at virgin state (t=0, all zero), then steps through
the digitized data points sorted by ascending |strain| so the loading is
monotonic from undamaged to fully-damaged state.

Re-run this script if modified_ds.csv changes:
    python make_pcm_reference_csvs.py
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
DIGITIZED = Path.home() / "projects/NEML2_V3/tutorial/mazars/modified_ds.csv"
POISSON = 0.2     # must match the elastic nu used in the scenario .i files

data = np.loadtxt(DIGITIZED, delimiter=",")
strains = data[:, 0]
stresses = data[:, 1]

mask_t = strains >= 0
mask_c = strains < 0

# Tension: sort by ascending strain (smallest positive to largest positive)
order_t = np.argsort(strains[mask_t])
eps_t = strains[mask_t][order_t]
sig_t = stresses[mask_t][order_t]

# Compression: sort by ascending magnitude (closest to zero first, most negative last)
order_c = np.argsort(np.abs(strains[mask_c]))
eps_c = strains[mask_c][order_c]
sig_c = stresses[mask_c][order_c]

print(f"Tension:     {len(eps_t)} points, strain range [0, {eps_t.max():.4e}]")
print(f"Compression: {len(eps_c)} points, strain range [0, {eps_c.min():.4e}]")
print()

def write_v3_csv(out_path: Path, axial_strains, axial_stresses) -> None:
    """Write a v3-format verification CSV with a virgin t=0 row prepended."""
    header = ["time",
              "strain_xx", "strain_yy", "strain_zz",
              "strain_yz", "strain_xz", "strain_xy",
              "stress_xx", "stress_yy", "stress_zz",
              "stress_yz", "stress_xz", "stress_xy"]

    rows = [[0.0] + [0.0] * 12]   # virgin state at t=0
    for t, (e, s) in enumerate(zip(axial_strains, axial_stresses), start=1):
        ezz = -POISSON * e
        rows.append([float(t),
                     e, ezz, ezz, 0.0, 0.0, 0.0,
                     s, 0.0, 0.0, 0.0, 0.0, 0.0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{x:.18e}" for x in row])
    print(f"  wrote {out_path.relative_to(HERE.parent.parent.parent.parent)}"
          f"  ({len(rows)} rows incl. virgin t=0)")

write_v3_csv(HERE / "pcm_2001_handbook_tension" / "pcm_2001_handbook_tension.csv",
             eps_t, sig_t)
write_v3_csv(HERE / "pcm_2001_handbook_compression" / "pcm_2001_handbook_compression.csv",
             eps_c, sig_c)
print("\nDone.")
