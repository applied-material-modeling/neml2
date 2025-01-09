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
import subprocess
import argparse
import numpy as np


def extract_time(output: str) -> float:
    t = output.splitlines()[2].strip().split(" ")[1]
    return float(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", help="Path to the runner")
    parser.add_argument(
        "codename",
        help="Benchmark problem's code name (folder names under this directory)",
    )
    parser.add_argument("device", help="Device on which to run the benchmark")
    parser.add_argument(
        "-s",
        "--stable-time",
        help="If the time of a single evaluation is shorter than this stable time, repeat the evaluation until the total elapsed time reaches this stable time.",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--max-time",
        help="Increase nbatch until the run time exceeds this maximum time (in milliseconds)",
        default=300000,
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results",
        help="Output folder to collect the results. A CSV file <codename>_<device>.csv will be written under the output directory.",
    )
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    if not outdir.is_dir():
        raise RuntimeError("output path {} is not a directory".format(args.output))

    runner = Path(args.runner)
    if not runner.exists():
        raise RuntimeError("runner does not exist at {}".format(args.runner))

    benchdir = Path(args.codename)
    if not benchdir.exists() or not benchdir.is_dir():
        raise RuntimeError("Benchmark with codename {} not found".format(args.codename))

    device = args.device
    stable_time = args.stable_time
    max_time = args.max_time
    input = benchdir / "model.i"

    outfile = outdir / "{}_{}.csv".format(args.codename, device)
    nbatch = 1
    mean_time = 0
    with open(outfile, "w", buffering=1) as stream:
        stream.write("nbatch,neval,mean,std\n")

        while mean_time < max_time:
            times = []
            total_time = 0
            command = [
                str(runner),
                "--time",
                str(input),
                "driver",
                "device={}".format(device),
                "nbatch={}".format(nbatch),
            ]

            while total_time < stable_time:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                time = extract_time(result.stdout)
                times.append(time)
                total_time += time

            times = np.array(times)
            neval = len(times)
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(
                "nbatch: {}, neval: {}, mean: {}, std: {}".format(
                    nbatch, neval, mean_time, std_time
                )
            )
            stream.write("{},{},{},{}\n".format(nbatch, neval, mean_time, std_time))

            nbatch *= 2
