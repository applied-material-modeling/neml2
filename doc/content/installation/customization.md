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
| NEML2_TOOLS           | ON, <u>OFF</u>          | Create targets for utility binaries               |
| NEML2_WORK_DISPATCHER | ON, <u>OFF</u>          | Enable work dispatcher                            |
| NEML2_JSON            | ON, <u>OFF</u>          | Enable JSON support                               |
| NEML2_CSV             | ON, <u>OFF</u>          | Enable CSV support                                |

Additional configuration options can be passed via command line using the `-DOPTION` or `-DOPTION=ON` format (see e.g., [cmake manual](https://cmake.org/cmake/help/latest/manual/cmake.1.html)).

## Configure presets

Since many configure options are available for customizing the build, it is sometimes challenging to keep track of them during the development workflow. CMake introduces the concept of [preset](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) to help manage common configurations.

NEML2 predefines configure presets serving different development purposes:
- cc: Export compile commands for static analysis and language servers.
- dev: This preset is best suited for developing the C++ backend and utility binaries. Compiler optimization is turned off, and debug symbols are enabled.
- coverage: Unit tests are built with coverage flags enabled. `gcov` or similar tools can be used to record code coverage data.
- tsan: Build with thread sanitizer flags to detect races in tests and tools.
- asan: Build with address sanitizer flags to detect memory errors in tests and tools.
- release: Production build for the C++ backend and utility binaries.
- profiling: Build utility binaries and additionally link against gperftools' CPU profiler.

The configure presets and their corresponding configure options are summarized below.

| preset                        | cc    | dev   | coverage | tsan            | asan             | release | profiling |
| :---------------------------- | :---- | :---- | :------- | :-------------- | :--------------- | :------ | :-------- |
| CMAKE_BUILD_TYPE              | Debug | Debug | Coverage | ThreadSanitizer | AddressSanitizer | Release | Profiling |
| CMAKE_EXPORT_COMPILE_COMMANDS | ON    | OFF   | OFF      | OFF             | OFF              | OFF     | OFF       |
| NEML2_PCH                     | OFF   | ON    | ON       | ON              | ON               | ON      | ON        |
| NEML2_TESTS                   | ON    | ON    | ON       | ON              | ON               | ON      | OFF       |
| NEML2_TOOLS                   | ON    | ON    | OFF      | ON              | ON               | ON      | ON        |
| NEML2_WORK_DISPATCHER         | ON    | ON    | ON       | ON              | ON               | ON      | ON        |
| NEML2_JSON                    | ON    | ON    | ON       | ON              | ON               | ON      | ON        |
| NEML2_CSV                     | ON    | ON    | ON       | ON              | ON               | ON      | ON        |

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
- dev: Tests and utility binaries for development
- tsan: Tests and utility binaries built with ThreadSanitizer
- asan: Tests and utility binaries built with AddressSanitizer
- coverage: C++ backend compiled with coverage flags
- release: Tests and utility binaries for release
- profiling: utility binaries with debug symbols linked against profiler

To use a build preset, use the `--preset` option on the command line, i.e.
