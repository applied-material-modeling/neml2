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

"""Standalone pyzag example: a NEML2 material model through the chunked solver.

Loads a NEML2 nonlinear material system (a Kocks-Mecking viscoplastic model with
Voce isotropic hardening under mixed strain/stress control) and integrates a
batch of uniaxial-strain histories -- spread across temperature and peak strain
-- in one shot with pyzag's chunked, step-vectorized Newton solver
(``neml2.pyzag.NEML2PyzagFactory`` + ``RecursiveNonlinearEquationSolver``). Prints
the wall time and saves a trajectory plot.

Run: ``python neml2_general.py`` (requires NEML2 built and importable).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

try:
    import neml2
    from neml2.pyzag import NEML2PyzagFactory
except ImportError:
    print("This example requires NEML2 (built + on PYTHONPATH). Skipping.")
    sys.exit(0)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyzag import chunktime, nonlinear

NCHUNK = 50
NBATCH = 4
NTIME = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STRAIN = 0.1
STRAIN_RATE = 1.0e-3
TEMPERATURE = 600.0
CONTROL = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

RTOL = 1.0e-8
ATOL = 1.0e-10

MODEL_PATH = str(Path(__file__).resolve().parent / "model.i")
EQ_SYS_NAME = "eq_sys"

torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)
torch.manual_seed(42)


def build_loading():
    """Uniaxial strain-controlled loading, spread across the batch.

    Axis-0 strain ramps linearly to a per-batch peak (0.5*MAX_STRAIN..MAX_STRAIN)
    with a per-batch temperature (TEMPERATURE +/- 200 K) so the batch members give
    visibly distinct curves; other components are stress-free. Returns
    (times, temperature, loading, control), each shaped (NTIME, NBATCH, ...).
    """
    control_vec = torch.tensor(CONTROL)
    temps = torch.linspace(TEMPERATURE - 200.0, TEMPERATURE + 200.0, NBATCH)
    max_strains = torch.linspace(0.5 * MAX_STRAIN, MAX_STRAIN, NBATCH)
    end_time = MAX_STRAIN / STRAIN_RATE

    tgrid = torch.linspace(0.0, end_time, NTIME)
    egrid = torch.linspace(0.0, 1.0, NTIME)

    times = tgrid.unsqueeze(-1).expand(NTIME, NBATCH).contiguous()
    temperature = temps.unsqueeze(0).expand(NTIME, NBATCH).contiguous()
    loading = torch.zeros(NTIME, NBATCH, 6)
    loading[..., 0] = egrid.unsqueeze(-1) * max_strains.unsqueeze(0)
    control = control_vec.expand(NTIME, NBATCH, 6).contiguous()
    return times, temperature, loading, control


def chunked_solve(factory, times, temperature, loading, control):
    """Solve the whole trajectory with pyzag's chunked step-vectorized solver."""
    raw = {
        "control": control,
        "mixed_control": loading,
        "temperature": temperature.unsqueeze(-1),
        "t": times.unsqueeze(-1),
    }
    forces = torch.cat([raw[name] for name in factory.fvars], dim=-1)
    state0 = torch.zeros(NBATCH, factory.nstate)
    solver = nonlinear.RecursiveNonlinearEquationSolver(
        factory,
        step_generator=nonlinear.StepGenerator(block_size=NCHUNK),
        predictor=nonlinear.PreviousStepsPredictor(),
        nonlinear_solver=chunktime.ChunkNewtonRaphson(rtol=RTOL, atol=ATOL),
    )
    with torch.no_grad():
        return nonlinear.solve(solver, state0, NTIME, forces)


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def plot_trajectory(times, chunked, out_path):
    """Plot the pyzag chunked trajectories, one line per batch member."""
    t_np = times[:, 0].detach().cpu().numpy()
    chk_np = chunked.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    comps = [(0, "mixed_state[0] (axial)"), (6, "equivalent_plastic_strain")]
    for ax, (idx, label) in zip(axes, comps):
        for b in range(NBATCH):
            ax.plot(t_np, chk_np[:, b, idx], "-", lw=1.5, label=f"batch {b}")
        ax.set_xlabel("time")
        ax.set_ylabel(label)
    axes[0].set_title("pyzag chunked solve")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    nsys = neml2.load_nonlinear_system(MODEL_PATH, EQ_SYS_NAME)
    nsys.model.to(device=DEVICE)
    factory = NEML2PyzagFactory(nsys, compile=True)

    times, temperature, loading, control = build_loading()

    print(f"model: {MODEL_PATH}")
    print(f"nbatch={NBATCH} ntime={NTIME} nchunk={NCHUNK} nstate={factory.nstate}")

    _sync()
    t0 = time.perf_counter()
    chunked = chunked_solve(factory, times, temperature, loading, control)
    _sync()
    t_chunked = time.perf_counter() - t0
    print(f"pyzag chunked (nchunk={NCHUNK}): {t_chunked:.4f} s")

    out_png = Path(__file__).resolve().parent / "neml2_general.png"
    plot_trajectory(times, chunked, out_png)
    print(f"trajectory plot saved to {out_png}")


if __name__ == "__main__":
    main()
