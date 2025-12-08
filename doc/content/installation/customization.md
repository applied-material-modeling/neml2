# Build Customization {#build-customization}

[TOC]

\note
Refer to the [cmake manual](https://cmake.org/cmake/help/latest/manual/cmake.1.html) for more CMake command line options. For more fine-grained control over the configure, build, and install commands, please refer to the [CMake User Interaction Guide](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html).

## Configure options

The configuration of NEML2 can be customized via a variety of high-level configure options. Commonly used configuration options are summarized below. Default options are <u>underlined</u>.

| Option                | Values (<u>default</u>) | Description                                       |
| :-------------------- | :---------------------- | :------------------------------------------------ |
| NEML2_PCH             | <u>ON</u>, OFF          | Use precompiled headers to accelerate compilation |
| NEML2_TESTS           | <u>ON</u>, OFF          | Master knob for including/excluding all tests     |
| NEML2_RUNNER          | ON, <u>OFF</u>          | Create a simple neml2 executable                  |
| NEML2_PYBIND          | ON, <u>OFF</u>          | Create the Python bindings target                 |
| NEML2_DOC             | ON, <u>OFF</u>          | Create the documentation target                   |
| NEML2_WORK_DISPATCHER | ON, <u>OFF</u>          | Enable work dispatcher                            |
| NEML2_JSON            | ON, <u>OFF</u>          | Enable JSON support                               |

Additional configuration options can be passed via command line using the `-DOPTION` or `-DOPTION=ON` format (see e.g., [cmake manual](https://cmake.org/cmake/help/latest/manual/cmake.1.html)).

## Configure presets

Since many configure options are available for customizing the build, it is sometimes challenging to keep track of them during the development workflow. CMake introduces the concept of [preset](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) to help manage common configurations.

NEML2 predefines six configure presets, serving different development purposes:
- dev: This preset is best suited for developing the C++ backend and Python bindings. Compiler optimization is turned off, and debug symbols are enabled. In addition, targets for locally generating the documentation (this website) are enabled.
- coverage: Unit tests are built with coverage flags enabled. `gcov` or similar tools can be used to record code coverage data.
- runner: The NEML2 runner is built with the highest level of compiler optimization. The runner is an executable that can be used to parse, evaluate, inspect, and diagnose NEML2 input files.
- tsan: Build the NEML2 Runner with thread sanitizer flags. The Runner can then be used to detect races.
- release: Build both the C++ backend and the Python package for production runs.
- profiling: Similar to runner, but additionally links the executable against gperftools' CPU profiler for profiling purposes.

The configure presets and their corresponding configure options are summarized below.

| preset                | dev   | coverage | runner  | tsan            | release        | profiling |
| :-------------------- | :---- | :------- | :------ | :-------------- | :------------- | :-------- |
| CMAKE_BUILD_TYPE      | Debug | Coverage | Release | ThreadSanitizer | RelWithDebInfo | Profiling |
| NEML2_PCH             | ON    | ON       | ON      | ON              | ON             | ON        |
| NEML2_TESTS           | ON    | ON       |         |                 |                |           |
| NEML2_RUNNER          |       |          | ON      | ON              | ON             | ON        |
| NEML2_PYBIND          | ON    |          |         |                 | ON             |           |
| NEML2_DOC             | ON    |          |         |                 |                |           |
| NEML2_WORK_DISPATCHER | ON    | ON       | ON      | ON              | ON             | ON        |
| NEML2_JSON            | ON    | ON       | ON      | ON              | ON             | ON        |

To select a specific configure preset, use the `--preset` option on the command line.

While the default presets should cover most of the development stages, it is sometimes necessary to override certain options. In general, there are three ways of overriding the preset:
- Command line options
- Environment variables
- [CMakeUserPresets.json](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)

For example, the following command
```
cmake --preset release -DNEML2_WORK_DISPATCHER=OFF -S .
```
would use the configure preset "release" while disabling the work dispatcher, and the same could be achieved via environment variables or user presets.

## Build presets

Once the project is configured (e.g., using configure presets), one or more build targets will be generated. Different configure options would generate different sets of build targets. The `--target` command line option can be used to specify the target to build. Similar to configure presets, build presets are used to pre-define "groups" of build targets.

NEML2 offers a number of build presets:
- dev-cpp: C++ backend with tests
- dev-python: Python bindings with tests
- dev-doc: HTML documentation
- coverage: C++ backend compiled with coverage flags
- runner: Runner
- release: C++ backend and Python bindings for release
- profiling: Runner with debug symbols linked against profiler

To use a build preset, use the `--preset` option on the command line.
