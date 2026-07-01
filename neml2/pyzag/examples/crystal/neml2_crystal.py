# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: pyzag
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

"""Multi-grain crystal plasticity (Taylor model) through pyzag's chunked solver.

A Taylor polycrystal aggregate of ``NGRAINS`` cubic crystals is loaded in mixed
control (axial strain rate prescribed, transverse stress free). The aggregate
shares one global deformation-rate / stress state; each grain integrates its own
elastic strain, orientation, and slip hardening.

The whole time history is solved in one shot by pyzag's chunked Newton-Raphson:
the NEML2 mixed-control equation system (loaded via
``neml2.load_nonlinear_system``) is bridged to pyzag with
``neml2.pyzag.NEML2PyzagFactory``, and the bidiagonal-in-time system is factorised
with the Thomas solver over chunks of ``NCHUNK`` steps. The macroscopic axial
stress-strain response is plotted.

``DEVICE`` defaults to CUDA when a GPU is available; the default device is set
globally before the NEML2 model is loaded.

Run:
    conda activate nemlv3_pyzag
    python neml2_crystal.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

NGRAINS = 1000
NBATCH = 1
NCHUNK = 10
NTIME = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)

AXIAL_STRAIN_RATE = 1.0e-3
RTOL = 1e-8
ATOL = 1e-10

_HERE = Path(__file__).resolve().parent
MIXED_I = str(_HERE / "model_mixed.i")

try:
    import neml2
    from neml2.pyzag import NEML2PyzagFactory
    from pyzag import chunktime, nonlinear
except ImportError as exc:
    print(
        "Skipping neml2_crystal example: NEML2 (with the pyzag interface) is "
        f"not importable in this environment.\n  ({exc})\n"
        "Activate the NEML2 environment (e.g. `conda activate nemlv3_pyzag`) "
        "and re-run.",
        file=sys.stderr,
    )
    sys.exit(0)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PER_GRAIN_BASE = 10
DENSE_SIZE = 12


def random_orientations(nbatch, ngrains, device):
    """Return ``(nbatch, ngrains, 3)`` uniform-on-SO(3) orientations as MRPs."""
    total = nbatch * ngrains
    u1 = torch.rand(total, device=device)
    u2 = torch.rand(total, device=device)
    u3 = torch.rand(total, device=device)
    sqrt_1mu1 = torch.sqrt(1.0 - u1)
    sqrt_u1 = torch.sqrt(u1)
    two_pi = 2.0 * math.pi
    qw = sqrt_1mu1 * torch.sin(two_pi * u2)
    qx = sqrt_1mu1 * torch.cos(two_pi * u2)
    qy = sqrt_u1 * torch.sin(two_pi * u3)
    qz = sqrt_u1 * torch.cos(two_pi * u3)
    sign = 1.0 - 2.0 * (qw < 0).to(qw.dtype)
    qw, qx, qy, qz = qw * sign, qx * sign, qy * sign, qz * sign
    inv = 1.0 / (1.0 + qw)
    flat = torch.stack([qx * inv, qy * inv, qz * inv], dim=-1)
    return flat.reshape(nbatch, ngrains, 3).contiguous()


def build_time_grid(ntime, device):
    """Geometric time ramp (small dt at start) to stay in the chunk-Newton basin."""
    t_max = (ntime - 1) * 0.5
    log_times = torch.logspace(-3, math.log10(t_max), ntime - 1, device=device, dtype=torch.float64)
    return torch.cat([torch.zeros(1, device=device, dtype=torch.float64), log_times])


def build_loading(ntime, device):
    """Mixed control: axial strain-rate prescribed, transverse stress free."""
    control = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    prescribed = torch.tensor(
        [AXIAL_STRAIN_RATE, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64
    )
    control_t = control.unsqueeze(0).expand(ntime, 6).contiguous()
    prescribed_t = prescribed.unsqueeze(0).expand(ntime, 6).contiguous()
    return control_t, prescribed_t


def extract_trajectory(result, ngrains):
    """Split the flat pyzag state into per-grain and global trajectories."""
    ntime, nbatch, nflat = result.shape
    expected = ngrains * PER_GRAIN_BASE + DENSE_SIZE
    if nflat != expected:
        raise ValueError(f"flat state {nflat} != expected {expected}")
    block = result[..., : ngrains * PER_GRAIN_BASE].reshape(ntime, nbatch, ngrains, PER_GRAIN_BASE)
    dense = result[..., ngrains * PER_GRAIN_BASE :]
    return {
        "elastic_strain": block[..., 0:6],
        "orientation": block[..., 6:9],
        "slip_hardening": block[..., 9],
        "deformation_rate": dense[..., :6],
        "cauchy_stress": dense[..., 6:12],
    }


def run_pyzag(orientations, times, control, prescribed):
    """Solve the whole mixed-control history in one pyzag chunked solve."""
    nbatch, ngrains = orientations.shape[:2]
    ntime = times.shape[0]
    device, dtype = orientations.device, orientations.dtype

    nsys = neml2.load_nonlinear_system(MIXED_I, "eq_sys")
    nsys.model.to(device=device)
    factory = NEML2PyzagFactory(nsys, compile=True)

    ic_dict = {
        "elastic_strain": torch.zeros(nbatch, ngrains, 6, device=device, dtype=dtype),
        "orientation": orientations,
        "slip_hardening": torch.zeros(nbatch, ngrains, device=device, dtype=dtype),
        "deformation_rate": torch.zeros(nbatch, 6, device=device, dtype=dtype),
        "target_cauchy_stress": torch.zeros(nbatch, 6, device=device, dtype=dtype),
    }
    y0 = factory.assemble_state(ic_dict, dynamic_dim=1)

    forces_dict = {
        "control": control.unsqueeze(1).expand(ntime, nbatch, 6).contiguous(),
        "prescribed": prescribed.unsqueeze(1).expand(ntime, nbatch, 6).contiguous(),
        "t": times.unsqueeze(-1).expand(ntime, nbatch).contiguous(),
        "vorticity": torch.zeros(ntime, nbatch, 3, device=device, dtype=dtype),
    }
    forces = factory.assemble_forces(forces_dict, dynamic_dim=2)

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        factory,
        step_generator=nonlinear.StepGenerator(NCHUNK),
        predictor=nonlinear.PreviousStepsPredictor(),
        direct_solve_operator=chunktime.BidiagonalThomasFactorization,
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=RTOL, atol=ATOL),
    )
    with torch.no_grad():
        result = nonlinear.solve(solver, y0, ntime, forces)
    return extract_trajectory(result, ngrains)


def _axial_stress_strain(times, traj):
    """Macroscopic axial (strain, stress) as ``(ntime, nbatch)`` numpy arrays."""
    t = times.detach().cpu().numpy()
    d0 = traj["deformation_rate"][..., 0].detach().cpu().numpy()
    dt = np.diff(t)
    inc = 0.5 * (d0[1:] + d0[:-1]) * dt[:, None]
    strain = np.concatenate([np.zeros((1, d0.shape[1])), np.cumsum(inc, axis=0)], axis=0)
    stress = traj["cauchy_stress"][..., 0].detach().cpu().numpy()
    return strain, stress


def plot_trajectory(times, pyzag_traj, out_path):
    """Plot the macroscopic axial stress-strain response, one line per batch."""
    strain, stress = _axial_stress_strain(times, pyzag_traj)
    nbatch = strain.shape[1]
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in range(nbatch):
        ax.plot(
            strain[:, b] * 100.0,
            stress[:, b],
            "-",
            lw=2,
            label="pyzag" if nbatch == 1 else f"pyzag b{b}",
        )
    ax.set_xlabel("macroscopic axial strain (%)")
    ax.set_ylabel("macroscopic axial stress")
    ax.set_title("Taylor polycrystal: macroscopic stress-strain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    print(
        f"config: NGRAINS={NGRAINS} NBATCH={NBATCH} NCHUNK={NCHUNK} "
        f"NTIME={NTIME} DEVICE={DEVICE}"
    )
    orientations = random_orientations(NBATCH, NGRAINS, DEVICE)
    times = build_time_grid(NTIME, DEVICE)
    control, prescribed = build_loading(NTIME, DEVICE)

    t0 = time.perf_counter()
    pyzag_traj = run_pyzag(orientations, times, control, prescribed)
    t_pyzag = time.perf_counter() - t0
    print(f"pyzag chunked solve: {t_pyzag:.3f} s")

    out_png = _HERE / "neml2_crystal.png"
    plot_trajectory(times, pyzag_traj, out_png)
    print(f"trajectory plot saved to {out_png}")


if __name__ == "__main__":
    main()
