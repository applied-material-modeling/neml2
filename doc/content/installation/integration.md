# External Project Integration {#external-project-integration}

[TOC]

## Makefile integration

NEML2 officially supports GNU Make integration via `pkg-config`.

After installing NEML2, point `PKG_CONFIG_PATH` to the directory containing `neml2.pc` (typically `<prefix>/share/pkgconfig`), then query compile and link flags from `pkg-config`.

The following example Makefile compiles a program named `foo` from `main.cxx`:

```make
CXX ?= c++
PKG_CONFIG ?= pkg-config

TARGET := foo
SOURCES := main.cxx
comma := ,

CXXFLAGS += $(shell $(PKG_CONFIG) --cflags neml2)
LDFLAGS += $(foreach libdir,$(shell $(PKG_CONFIG) --libs-only-L neml2),$(patsubst -L%,-Wl$(comma)-rpath$(comma)%,$(libdir)))
LDLIBS += $(shell $(PKG_CONFIG) --libs neml2)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(LDLIBS)

clean:
	rm -f $(TARGET)
```

Example usage:

```bash
PKG_CONFIG_PATH=/path/to/neml2-install/share/pkgconfig make
./foo
```

## CMake integration

### Sub-directory

Integrating NEML2 into a project that already uses CMake is fairly straightforward. The following CMakeLists.txt snippet links NEML2 into the target executable called `foo`:

```
add_subdirectory(neml2)
add_executable(foo main.cxx)
target_link_libraries(foo neml2)
```

The above snippet assumes NEML2 is checked out to the directory %neml2, i.e., as a git submodule.

### FetchContent

Alternatively, you may use CMake's `FetchContent` module to integrate NEML2 into your project:

```
FetchContent_Declare(
  neml2
  GIT_REPOSITORY https://github.com/applied-material-modeling/neml2.git
  GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(neml2)
add_executable(foo main.cxx)
target_link_libraries(foo neml2)
```

### find_package

NEML2 can also be discovered from its installation location relying on CMake's config mode `find_package` function

```
find_package(neml2 CONFIG)
add_executable(foo main.cxx)
target_link_libraries(foo neml2::neml2)
```

Note that the config mode search defines several imported targets under the `neml2::` namespace. For example, `neml2::misc` corresponds to the `misc` library of NEML2, etc. The `neml2::neml2` is an interface target that transitively links to all NEML2 libraries.
