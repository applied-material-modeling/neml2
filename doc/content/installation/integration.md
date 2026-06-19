(external-project-integration)=
# C++ integration

The PyPI wheel bundles every artifact a downstream C++ project needs —
shared libraries, public headers, CMake config exports, and pkg-config
files. Pointing your build system at the wheel's install tree is enough;
no second NEML2 install or separate libtorch download is required —
the `torch` wheel that pip installs alongside neml2 provides libtorch,
and NEML2's exported config wires the two together.

This page assumes NEML2 was installed with `pip install neml2`. The
relevant subdirectories are all under

```
<site-packages>/neml2/
├── include/
├── lib/
└── share/
    ├── cmake/neml2/
    └── pkgconfig/
```

:::{important}
Every shell snippet on this page assumes a `$NEML2_ROOT` variable
pointing at the wheel root. Set it once per shell:

```shell
NEML2_ROOT=$(python -c "import neml2, os; print(os.path.dirname(neml2.__file__))")
```
:::

## CMake

NEML2 ships a CMake config package. After installing the wheel, point
CMake at the wheel root via `CMAKE_PREFIX_PATH` (or directly at the
config dir via `neml2_DIR`) and use `find_package`:

```cmake
find_package(neml2 CONFIG REQUIRED)

add_executable(foo main.cxx)
target_link_libraries(foo PRIVATE neml2::aoti)
```

Build it with either of:

```shell
NEML2_ROOT=$(python -c "import neml2, os; print(os.path.dirname(neml2.__file__))")

# Point at the wheel root — cmake searches the standard share/cmake/<pkg>/ subtree.
cmake -B build -DCMAKE_PREFIX_PATH=$NEML2_ROOT -S .

# Or point neml2_DIR straight at the config directory.
cmake -B build -Dneml2_DIR=$NEML2_ROOT/share/cmake/neml2 -S .

cmake --build build -j$(nproc)
```

The `neml2::aoti` target carries the NEML2 runtime library and
propagates its include directories and link flags. Torch is not bundled
into the NEML2 install — it comes from the `torch` wheel pip installs
alongside neml2, and is discovered transitively via the exported
config.

## pkg-config

The same install also ships pkg-config files for build systems that
don't speak CMake (plain Makefiles, Meson, Autotools, …).

```shell
NEML2_ROOT=$(python -c "import neml2, os; print(os.path.dirname(neml2.__file__))")
export PKG_CONFIG_PATH=$NEML2_ROOT/share/pkgconfig:$PKG_CONFIG_PATH
pkg-config --cflags --libs neml2
```

A minimal Makefile that compiles `foo` from `main.cxx`:

```make
CXX ?= c++
PKG_CONFIG ?= pkg-config

TARGET := foo
SOURCES := main.cxx
comma := ,

CXXFLAGS += $(shell $(PKG_CONFIG) --cflags neml2)
# Add an rpath entry for every -L the .pc file emitted so the resulting
# binary can find the bundled libneml2.so + libtorch at runtime
# without LD_LIBRARY_PATH.
LDFLAGS  += $(foreach libdir,$(shell $(PKG_CONFIG) --libs-only-L neml2),$(patsubst -L%,-Wl$(comma)-rpath$(comma)%,$(libdir)))
LDLIBS   += $(shell $(PKG_CONFIG) --libs neml2)

.PHONY: all clean
all: $(TARGET)
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(LDLIBS)
clean:
	rm -f $(TARGET)
```

Build:

```shell
PKG_CONFIG_PATH=$NEML2_ROOT/share/pkgconfig make
./foo
```

The pkg-config files NEML2 ships are:

- `neml2.pc` — meta entry. `pkg-config --cflags --libs neml2` pulls in
  everything (core, torch, nlohmann_json). Use this unless you have a
  specific reason not to.
- `neml2-core.pc` — just `libneml2` and its public headers. Useful
  if your build system already manages torch / nlohmann_json
  independently.
- `neml2-torch.pc`, `neml2-nlohmann-json.pc` — the bundled dependency
  fragments that `neml2.pc` composes.

## Header includes

C++ source includes are namespaced under `neml2/csrc/`:

```cpp
#include "neml2/csrc/aoti/Model.h"
```

This mirrors the layout under `<site-packages>/neml2/include/`, so the
same source compiles whether NEML2 was discovered via CMake's
`find_package` or via `pkg-config --cflags`.

## Runtime library lookup

The bundled `libneml2.so` lives under `<site-packages>/neml2/lib/`
and links against the libtorch shipped in the sibling `torch` wheel at
`<site-packages>/torch/lib/`, reached via an `$ORIGIN/../../torch/lib`
rpath baked into NEML2's library. If you set rpath at link time (CMake
does this by default; the Makefile snippet above does it explicitly)
the resulting binary runs out of the box. If you opt out of rpath, set
`LD_LIBRARY_PATH=$NEML2_ROOT/lib:$NEML2_ROOT/../torch/lib` (Linux) or
`DYLD_LIBRARY_PATH` (macOS) before running.
