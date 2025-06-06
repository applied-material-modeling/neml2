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
import sys
import json
import subprocess
import argparse
import concurrent.futures
import itertools
import multiprocessing


def batch(alist, n=1):
    l = len(alist)
    for ndx in range(0, l, n):
        yield alist[ndx : min(ndx + n, l)]


def filter_compile_commands(compile_commands, srcs, headers, include_dirs):
    included = lambda f: any(f.startswith(d) for d in include_dirs)
    filtered = []
    for cmd in compile_commands:
        if not included(cmd["file"]):
            continue
        if cmd["file"] in srcs:
            filtered.append(cmd)
            print(cmd["file"])
        else:
            new_cmd = cmd["command"].split()
            oidx = new_cmd.index("-o")
            new_cmd.pop(oidx)
            new_cmd.pop(oidx)
            cidx = new_cmd.index("-c")
            new_cmd[cidx] = "-E"
            new_cmd.insert(cidx, "-MM")
            deps = subprocess.run(
                new_cmd, capture_output=True, text=True, check=True, cwd=original.parent
            ).stdout.splitlines()[1:]
            deps = [d.replace(" \\", "").strip() for d in deps]
            if any(dep.endswith(h) for h in headers for dep in deps):
                filtered.append(cmd)
                print(cmd["file"])
    return filtered


if __name__ == "__main__":
    # cliargs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--compile-commands",
        type=str,
        default="compile_commands.json",
        help="Path to compile_commands.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="compile_commands_filtered.json",
        help="Path to output file",
    )
    parser.add_argument(
        "-d",
        "--diff",
        type=str,
        nargs=2,
        default=["HEAD", "origin/main"],
        help="Only include compile commands for files that are different between two SHAs.",
    )
    parser.add_argument(
        "-i",
        "--include-dirs",
        type=str,
        default="src/neml2",
        help="A comma-separated list of directories. Only include compile commands for files in these directories. Specify these directories relative to the root of the repository.",
    )

    # parse cliargs
    args = parser.parse_args()
    original = Path(args.compile_commands)
    if not original.exists():
        print("Error: {} does not exist.".format(original))
        sys.exit(1)
    output = Path(args.output)
    sha1 = args.diff[0]
    sha2 = args.diff[1]
    root = Path(__file__).parent.parent
    include_dirs = args.include_dirs.split(",")
    include_dirs = [str((root / d).resolve()) for d in include_dirs]

    # get list of files that are different between two SHAs
    diff_files = subprocess.run(
        ["git", "diff", "--name-only", sha1, sha2], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    headers = [str((root / f).resolve()) for f in diff_files if f.endswith(".h")]
    srcs = [str((root / f).resolve()) for f in diff_files if f.endswith(".cxx")]
    print("Modified headers:")
    print("\n".join(headers), "\n")
    print("Modified sources:")
    print("\n".join(srcs), "\n")

    # read original compile_commands.json
    with open(original) as f:
        compile_commands = json.load(f)

    # filter compile_commands.json
    print("Files to be included in compile_commands.json:")
    executor = concurrent.futures.ProcessPoolExecutor(multiprocessing.cpu_count())
    futures = [
        executor.submit(filter_compile_commands, group, srcs, headers, include_dirs)
        for group in batch(compile_commands, 10)
    ]
    concurrent.futures.wait(futures)
    filtered = list(itertools.chain.from_iterable([f.result() for f in futures]))

    # write filtered compile_commands.json
    print("\nFiltered files {:d}/{:d}".format(len(filtered), len(compile_commands)))
    with open(output, "w") as f:
        json.dump(filtered, f, indent=2)
