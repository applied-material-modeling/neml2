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

"""Minimal, self-contained crystal-plasticity material calibration with pyzag.

Shows the full pyzag adjoint-calibration pattern end to end:

    1. Load a mixed-control Taylor crystal-plasticity NEML2 model and wrap it in
       ``neml2.pyzag.NEML2PyzagFactory`` (BLOCK+DENSE multi-group path).
    2. Generate a *synthetic* experiment in-script -- random grain orientations
       plus a strain/time grid built from ``MAX_STRAIN``/``RATE``/``NTIME`` --
       so no external dataset is needed.
    3. Compute a ground-truth axial stress-strain target at the model's default
       hardening parameters, then perturb those parameters away from truth.
    4. Recover them with an LBFGS loop that differentiates the chunked solve via
       ``pyzag.nonlinear.solve_adjoint`` (analytic parameter gradients), with a
       ``RangeRescale`` reparametrization so the optimizer works in scaled space.
    5. Assert the loss decreases; save a stress-strain plot and a results JSON.

Only the three Voce/slip hardening parameters are calibrated:

    slip_strength_constant_strength     (initial slip resistance, MPa)
    voce_hardening_initial_slope        (initial hardening slope, MPa)
    voce_hardening_saturated_hardening  (saturated hardening, MPa)

``DEVICE`` defaults to CUDA when available; ``torch.set_default_device`` is set
before the model is loaded so its ``[Tensors]`` constants land on the target
device. The factory uses ``compile=True`` (the first run pays a one-time compile
cost). Committed defaults (200 grains, 100 steps, 5 LBFGS iters) run in a few
minutes on an idle GPU; for a fast smoke check drop to NGRAINS=10, NTIME=30,
N_ITER=3.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    import neml2
    import neml2.types
    from neml2.pyzag import NEML2PyzagFactory
except ImportError:
    print(
        "SKIP: this example requires the NEML2 Python bindings with the "
        "pyzag backend (import neml2; from neml2.pyzag import NEML2PyzagFactory).\n"
        "Install/build NEML2 with pyzag support and re-run."
    )
    raise SystemExit(0)

from pyzag import chunktime, nonlinear, reparametrization

NGRAINS = 200
NCHUNK = 5
NTIME = 100
N_ITER = 5
MAX_STRAIN = 0.02
RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MODEL_FILE = Path(__file__).resolve().parent / "model_mixed_calibration.i"
PLOT_FILE = Path(__file__).resolve().parent / "calibration_result.png"
JSON_FILE = Path(__file__).resolve().parent / "calibration_result.json"

CALIBRATION_PARAMS = [
    "slip_strength_constant_strength",
    "voce_hardening_initial_slope",
    "voce_hardening_saturated_hardening",
]

PARAM_RANGES = {
    "slip_strength_constant_strength": (1.0, 2_000.0),
    "voce_hardening_initial_slope": (1e-3, 50_000.0),
    "voce_hardening_saturated_hardening": (1.0, 2_000.0),
}

PERTURB = {
    "slip_strength_constant_strength": 0.75,
    "voce_hardening_initial_slope": 2.0,
    "voce_hardening_saturated_hardening": 0.70,
}

_PER_GRAIN_BASE = 10
_DENSE_BASE = 12
_AXIAL = 2


def random_orientations(ngrains, dtype, device):
    """Random uniform SO(3) grain orientations as ``(ngrains, 3)`` MRPs."""
    g = torch.randn(ngrains, 3, 3, dtype=dtype, device=device)
    q, r = torch.linalg.qr(g)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    q[:, :, 0] = q[:, :, 0] * det.unsqueeze(-1)
    r2 = neml2.types.R2(q, 1)
    return neml2.types.MRP.from_matrix(r2).data.to(dtype=dtype, device=device).contiguous()


def strain_time_grid(max_strain, ntime, rate, dtype, device):
    """Uniform strain grid and the matching time grid for a constant strain rate."""
    strain = torch.linspace(0.0, max_strain, ntime, dtype=dtype, device=device)
    return strain, strain / rate


def build_factory(device):
    """Load the NEML2 system, exposing only CALIBRATION_PARAMS as torch params."""
    nsys = neml2.load_nonlinear_system(str(MODEL_FILE), "eq_sys")
    return NEML2PyzagFactory(nsys, include_parameters=CALIBRATION_PARAMS, compile=True)


class CalibrationModule(torch.nn.Module):
    """Differentiable module returning the predicted axial aggregate stress."""

    def __init__(self, factory, orientations, strain, times, nchunk):
        super().__init__()
        self.factory = factory
        self.nchunk = nchunk

        self.register_buffer("orientations", orientations)
        self.register_buffer("strain", strain)

        ngrains = orientations.shape[0]
        ntime = strain.shape[0]
        device = orientations.device
        dtype = orientations.dtype
        nbatch = 1

        ic_dict = {
            "elastic_strain": torch.zeros(nbatch, ngrains, 6, device=device, dtype=dtype),
            "orientation": orientations.unsqueeze(0),
            "slip_hardening": torch.zeros(nbatch, ngrains, device=device, dtype=dtype),
            "deformation_rate": torch.zeros(nbatch, 6, device=device, dtype=dtype),
            "target_cauchy_stress": torch.zeros(nbatch, 6, device=device, dtype=dtype),
        }
        self._y0 = self.factory.assemble_state(ic_dict, dynamic_dim=1)

        control = torch.zeros(ntime, nbatch, 6, device=device, dtype=dtype)
        control[..., _AXIAL] = 1.0
        prescribed = torch.zeros(ntime, nbatch, 6, device=device, dtype=dtype)
        prescribed[..., _AXIAL] = RATE
        t = times.reshape(ntime, 1).expand(ntime, nbatch).contiguous()
        vorticity = torch.zeros(ntime, nbatch, 3, device=device, dtype=dtype)
        self._forces = self.factory.assemble_forces(
            {
                "control": control,
                "prescribed": prescribed,
                "t": t,
                "vorticity": vorticity,
            },
            dynamic_dim=2,
        )
        self._ntime = ntime

    @property
    def calibration_parameters(self):
        """The calibration ``nn.Parameter`` references (reparametrization-aware)."""
        out = []
        for name in CALIBRATION_PARAMS:
            parametrizations = getattr(self.factory, "parametrizations", None)
            if parametrizations is not None and name in parametrizations:
                out.append(parametrizations[name].original)
            else:
                out.append(getattr(self.factory, name))
        return out

    def forward(self):
        """Predicted axial aggregate stress, shape ``(ntime,)``."""
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.factory,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
            direct_solve_operator=chunktime.BidiagonalThomasFactorization,
            nonlinear_solver=chunktime.ChunkNewtonRaphsonLineSearch(
                rtol=1e-6, atol=1e-8, linesearch_iter=5
            ),
        )
        result = nonlinear.solve_adjoint(solver, self._y0, self._ntime, self._forces)
        return result[..., -6 + _AXIAL].squeeze(-1)


def apply_reparametrization(module):
    """Wrap each calibration parameter in RangeRescale for scaled-space LBFGS."""
    device = module.orientations.device
    dtype = module.orientations.dtype
    map_dict = {
        f"factory.{name}": reparametrization.RangeRescale(
            torch.tensor(lo, device=device, dtype=dtype),
            torch.tensor(hi, device=device, dtype=dtype),
            clamp=True,
        )
        for name, (lo, hi) in PARAM_RANGES.items()
    }
    reparam = reparametrization.Reparameterizer(map_dict, error_not_provided=False)
    reparam(module)
    return reparam


def calibrate(module, target, n_iter):
    """Run LBFGS (strong-Wolfe line search); return the loss history."""
    optimizer = torch.optim.LBFGS(
        module.calibration_parameters,
        lr=1.0,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )
    history = []

    def closure():
        optimizer.zero_grad()
        pred = module()
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        return loss

    for it in range(n_iter):
        loss = optimizer.step(closure)
        loss_v = float(loss.detach())
        history.append(loss_v)
        print(f"  iter {it + 1:2d}  loss = {loss_v:.6e}")
    return history


def save_plot(strain, target, pred_init, pred_final):
    """Save a target vs initial-guess vs calibrated stress-strain overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = strain.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s, target.detach().cpu().numpy(), "k-", lw=2, label="synthetic target")
    ax.plot(s, pred_init.detach().cpu().numpy(), "r--", label="initial guess")
    ax.plot(s, pred_final.detach().cpu().numpy(), "b:", lw=2, label="calibrated")
    ax.set_xlabel("axial strain (mm/mm)")
    ax.set_ylabel("axial stress (MPa)")
    ax.set_title("Crystal-plasticity calibration (pyzag adjoint)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float64)
    device = torch.device(DEVICE)
    if device.type == "cuda":
        torch.set_default_device(device)
    print(f"Running on {device}")

    orientations = random_orientations(NGRAINS, torch.float64, device)
    strain, times = strain_time_grid(MAX_STRAIN, NTIME, RATE, torch.float64, device)

    factory = build_factory(device)
    module = CalibrationModule(factory, orientations, strain, times, NCHUNK).to(device)

    truth = {n: float(getattr(factory, n).detach()) for n in CALIBRATION_PARAMS}
    print(f"Ground-truth parameters: {truth}")
    with torch.no_grad():
        target = module()

    with torch.no_grad():
        for name, factor in PERTURB.items():
            getattr(factory, name).mul_(factor)
        factory._update_parameter_values()
    perturbed = {n: float(getattr(factory, n).detach()) for n in CALIBRATION_PARAMS}
    print(f"Perturbed (starting) parameters: {perturbed}")

    with torch.no_grad():
        pred_init = module()

    apply_reparametrization(module)

    print(f"Calibrating {NGRAINS} grains, {NTIME} steps, {N_ITER} LBFGS iters...")
    history = calibrate(module, target, N_ITER)

    with torch.no_grad():
        pred_final = module()
    recovered = {n: float(getattr(factory, n).detach()) for n in CALIBRATION_PARAMS}
    print(f"Recovered parameters:  {recovered}")

    assert history[-1] < history[0], "loss did not decrease"
    assert torch.isfinite(pred_final).all(), "prediction contains non-finite values"
    print(f"Loss decreased {history[0]:.6e} -> {history[-1]:.6e}")

    results = {
        "ground_truth": truth,
        "initial": perturbed,
        "final": recovered,
        "loss_initial": history[0],
        "loss_final": history[-1],
        "iters_done": len(history),
        "loss_history": history,
    }
    with JSON_FILE.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))

    save_plot(strain, target, pred_init, pred_final)
    print(f"Saved plot to {PLOT_FILE}")
    print(f"Saved results to {JSON_FILE}")


if __name__ == "__main__":
    main()
