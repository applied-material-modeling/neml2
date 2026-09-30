#!/usr/bin/env python3

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

"""Verification: NEML2 power-law creep vs the original MOOSE x447 fuel simulation.
Compares average von Mises stress (and temperature, burnup) over the full 1.79-yr
run, row-by-row (identical time grids). No re-run -- reads the two output CSVs."""

import csv
import statistics

mo = list(csv.DictReader(open("moose_x447_fuel.csv")))
ne = list(csv.DictReader(open("neml2_fuel.csv")))
n = min(len(mo), len(ne))
t = [float(r["time"]) for r in mo]
vm_mo = [float(r["vonmises_avg"]) for r in mo]
vm_ne = [float(r["vonmises_avg"]) for r in ne]
peak = max(abs(v) for i, v in enumerate(vm_mo) if t[i] > 1e4)
rels = sorted(abs(vm_mo[i] - vm_ne[i]) / peak for i in range(1, n) if t[i] > 1e4)


def pct(p):
    return rels[int(p * len(rels))]


print(f"NEML2 vs MOOSE (x447), {n - 1} steps, {t[-1] / 3.156e7:.2f} yr, identical grids")
print(
    f"von Mises (avg), rel-to-peak:  median={statistics.median(rels):.2e}  "
    f"90th={pct(0.90):.2e}  99th={pct(0.99):.2e}  max={max(rels):.2e}"
)
for col, lab in [("temp_fuel_avg", "temperature"), ("burnup_avg", "burnup")]:
    a = [float(r[col]) for r in mo]
    b = [float(r[col]) for r in ne]
    sc = max(abs(x) for x in a) or 1.0
    print(f"{lab:12s} rel: {max(abs(a[i] - b[i]) for i in range(n)) / sc:.2e}")
