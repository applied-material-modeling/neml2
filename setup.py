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

import os
import re
import subprocess
import sys
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Give up on windows
        if self.compiler.compiler_type == "msvc":
            raise RuntimeError("MSVC not supported")

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve() / "neml2"

        # Influential env vars
        additional_configure_args = os.environ.get("CMAKE_ARGS", "").split()
        njob = os.environ.get("CMAKE_BUILD_JOBS", None)

        # Configure arguments
        configure_args = [
            "--fresh",
            "-GNinja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DNEML2_PYBIND=ON",
            "-DNEML2_TESTS=OFF",
            "-DNEML2_RUNNER=OFF",
            "-DNEML2_DOC=OFF",
            "-DNEML2_WHEEL=ON",
            "-S{}".format(ext.sourcedir),
            *additional_configure_args,
        ]
        if njob:
            configure_args += ["-DNEML2_CONTRIB_PARALLEL={}".format(njob)]

        # Build arguments
        build_args = ["--build", ".", "--target", "pyneml2"]
        if njob:
            build_args += ["-j", njob]

        # Install arguments
        install_args = ["--install", ".", "--prefix", extdir]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                configure_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Ensure that the build directory exists
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Clean the extension directory
        shutil.rmtree(extdir, ignore_errors=True)

        subprocess.run(["cmake", *configure_args], cwd=build_temp, check=True)
        subprocess.run(["cmake", *build_args], cwd=build_temp, check=True)
        subprocess.run(["cmake", *install_args], cwd=build_temp, check=True)


setup(
    ext_modules=[CMakeExtension("neml2")],
    cmdclass={"build_ext": CMakeBuild},
)
