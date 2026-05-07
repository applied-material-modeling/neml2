# Purpose

NEML2 is a fast, vectorized library for developing, fitting, testing, and running material models. It is designed to be used in a variety of contexts, including:
- As a standalone library for developing and testing material models.
- As a library that can be called from other programming languages, currently only Python.
- Through the python interface and with the pyzag library, as an efficient way to train material models on large datasets.
- As a library that can be called from finite element codes via C++ or Python.

Our goal is to make it easy for users to develop and test material models, and to make it easy for users to use those models in a variety of contexts. We also want to make it easy for users to share their models with others, and to contribute to the development of the library itself.

# Contributing

All C++ code should be formatted using clang-format. The .clang-format file in the root of the repository specifies the formatting style. You can run clang-format on a file using the following command:
```clang-format -i <file>
```

All Python code should be formatted using black. You can run black on a file using the following command:
```black <file>
```

Each new feature should have corresponding unit tests.  This includes new basic functionalities, new tensor operations, and, especially, new models.  These should check for correctness in the model and, at least, the first derivatives.  Second derivative information is also required for some models and should be included in the tests when it's needed.  Unit tests live in `tests/unit`.  There are many examples of model unit tests in `tests/unit/models`.

If you contribute a entirely new type of model you should also include regression and verification tests.  Regression tests demonstrate common modeling workflows and ensure that they continue to work as the library evolves.  Verification tests compare the results of the model to known solutions, either from analytical solutions or from other software.  These tests live in `tests/regression` and `tests/verification`, respectively.

# Compiling and testing

NEML2 has several dependencies, most of which will be automatically downloaded and built.  A main dependency is `pytorch`, particularly the C++ API, which is used for the tensor operations.  While the build system can obtain libtorch, generally users are going to have a system/CUDA specific installed version.  This may mean you need to activate a conda environment prior to building the library. 

The library uses CMake as its build system.  Commonly, during development, you will want to build the library and run the unit tests.  You can do this using the following commands from the root of the repository:
```bash
cmake --preset dev -S . -B build/dev
cmake --build -j 8 --preset dev -B build/dev
```

This will make test binaries called `build/dev/tests/unit/unit_tests`, `build/dev/tests/regression/regression_tests`, and `build/dev/tests/verification/verification_tests`.  You can run these directly, they are built with CTest and the help messages will explain the various options.

# Workflow

Generally, the workflow for developing a new model will be as follows:
1. Develop the model based on the user's requests.  You can use an existing model as a template, but be sure to change the class name and the file names.  You should also add a new unit test for the model in `tests/unit/models`.
2. Ensure the library builds correctly and passes the unit tests.  NOTE: in the submodel development/build/test loop there isn't any point in running the regression and verification tests, so you can just run the unit tests.

This loop will be repeated to build up several smaller submodules.  Once the model is complete, you should add regression and verification tests.  You should also add documentation for the model, which will be built using Doxygen.  The documentation should include a description of the model, its parameters, and any relevant equations or references.