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

import sys
import importlib.metadata
from pathlib import Path


def show_missing_reqs(missing_reqs):
    print("-" * 79)
    print("There are missing Python package dependencies:")
    for missing_req in missing_reqs:
        print("  {}".format(missing_req))
    print("They can be installed using `pip install -r requirements.txt`.")
    print("-" * 79)


try:
    from packaging.requirements import Requirement
except:
    show_missing_reqs(["packaging"])


def get_reqs(path):
    """
    Recursively parse the requirements.txt to gather all dependencies
    """
    reqs = []
    with open(path, "r") as f:
        for req in f:
            req = req.strip()
            if req.startswith("#"):
                continue
            if req.startswith("-r"):
                reqfile = req.split(" ")[1]
                reqs += get_reqs(path.parent / reqfile)
            else:
                reqs.append(Requirement(req))
    return reqs


def _yield_missing_reqs(req: Requirement, current_extra: str = ""):
    if req.marker and not req.marker.evaluate({"extra": current_extra}):
        return

    try:
        version = importlib.metadata.distribution(req.name).version
    except importlib.metadata.PackageNotFoundError:  # req not installed
        yield req
    else:
        if req.specifier.contains(version):
            for child_req in importlib.metadata.metadata(req.name).get_all("Requires-Dist") or []:
                child_req_obj = Requirement(child_req)

                need_check, ext = False, None
                for extra in req.extras:
                    if child_req_obj.marker and child_req_obj.marker.evaluate({"extra": extra}):
                        need_check = True
                        ext = extra
                        break

                if need_check:  # check for extra reqs
                    yield from _yield_missing_reqs(child_req_obj, ext)

        else:  # main version not match
            yield req


if __name__ == "__main__":
    reqfile = Path(sys.argv[1])

    missing_reqs = []
    for req in get_reqs(reqfile):
        missing_reqs += [missing_req for missing_req in _yield_missing_reqs(req)]

    if missing_reqs:
        show_missing_reqs(missing_reqs)
        exit(1)
