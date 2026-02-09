#!/usr/bin/env python

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

from pathlib import Path
import argparse
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

color = {"cpu": "tab:red", "cuda": "tab:blue"}

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rcParams["text.usetex"] = True
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def macaulay(x, sharpness=100):
    return (0.5 + 0.5 * np.tanh(sharpness * x)) * x


def f(N, Ns, t0):
    return t0 / Ns * macaulay(N - Ns) + t0


def make_plot(ax, datas):
    Nmin = None
    Nmax = None
    params = {}
    for device, data in datas.items():
        N = data["nbatch"]
        t = data["mean"]
        ax.plot(
            N,
            t / 1000,
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=color[device],
            label=device,
        )
        p, _ = sp.optimize.curve_fit(f, N, t)
        params[device] = p
        Nmin = min(Nmin, np.min(N)) if Nmin else np.min(N)
        Nmax = max(Nmax, np.max(N)) if Nmax else np.max(N)

    N_ = np.logspace(np.log10(Nmin), np.log10(Nmax), 100)

    for device, p in params.items():
        t_ = f(N_, *p)
        ax.plot(
            N_,
            t_ / 1000,
            color=color[device],
            linestyle="--",
            label="{} (fitted)".format(device),
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Wall time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    return params


def find_critical_batch_size(data, device1, device2):
    Ns1 = data["{}_Ns".format(device1)]
    t01 = data["{}_t0".format(device1)]
    Ns2 = data["{}_Ns".format(device2)]
    t02 = data["{}_t0".format(device2)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("devices", nargs="+", help="Device on which to run the benchmark")
    parser.add_argument(
        "-o",
        "--output",
        default="results",
        help="Output folder to collect the results. A CSV file <codename>_<device>.csv must be present under the output directory for each given device.",
    )
    parser.add_argument(
        "-c",
        "--compare",
        nargs=2,
        help="The pair of devices to be compared",
    )
    args = parser.parse_args()

    outdir = Path(args.output)
    if not outdir.is_dir():
        raise RuntimeError("output path {} is not a directory".format(args.output))
    if not outdir.exists():
        raise RuntimeError("output path {} does not exist".format(args.output))

    # Collect benchmark codenames
    codenames = []
    for filename in outdir.glob("*.csv"):
        if filename.stem == "summary":
            continue
        codenames.append(filename.stem.split("_")[0])
    codenames = list(set(codenames))

    summary = {"condename": codenames}
    summary |= {"{}_Ns".format(device): [] for device in args.devices}
    summary |= {"{}_t0".format(device): [] for device in args.devices}
    for codename in codenames:
        fig, ax = plt.subplots()
        datas = {}
        for device in args.devices:
            datas[device] = pd.read_csv(outdir / "{}_{}.csv".format(codename, device))
        params = make_plot(ax, datas)
        fig.tight_layout()
        fig.savefig(outdir / "{}.pdf".format(codename))
        plt.close()

        for device, p in params.items():
            summary["{}_Ns".format(device)].append(p[0])
            summary["{}_t0".format(device)].append(p[1])

    if args.compare:
        devices = args.compare
        Ns1 = np.array(summary["{}_Ns".format(devices[0])])
        t01 = np.array(summary["{}_t0".format(devices[0])])
        Ns2 = np.array(summary["{}_Ns".format(devices[1])])
        t02 = np.array(summary["{}_t0".format(devices[1])])
        summary["Nc"] = Ns1 * t02 / t01
        summary["r0"] = t01 / t02
        summary["rinf"] = t01 / t02 * Ns2 / Ns1

    summary = pd.DataFrame(summary)
    summary.to_csv(outdir / "summary.csv", index=False)
