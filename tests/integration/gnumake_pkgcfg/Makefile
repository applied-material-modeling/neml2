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
