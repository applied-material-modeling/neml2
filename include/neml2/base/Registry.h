// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <memory>
#include <map>
#include <filesystem>

#include "neml2/misc/string_utils.h"
#include "neml2/base/OptionSet.h"

namespace neml2
{
/// Add a NEML2Object to the registry.  classname is the (unquoted)
/// c++ class.  Each object/class should only be registered once.
#define register_NEML2_object(classname)                                                           \
  static const char dummyvar_for_registering_obj_##classname = Registry::add<classname>(#classname)

/// Add a NEML2Object to the registry and associate it with a given name.  classname is the
/// (unquoted) c++ class.  Each object/class should only be registered once.
#define register_NEML2_object_alias(classname, registryname)                                       \
  static const char dummyvar_for_registering_obj_##classname =                                     \
      Registry::add<classname>(registryname)

class NEML2Object;

using BuildPtr = std::shared_ptr<NEML2Object> (*)(const OptionSet & options);

/// Information on a NEML2Object that is registered in the Registry.
struct NEML2ObjectInfo
{
  /// The object type demangled from the C++ typeid
  std::string type_name;
  /// Expected options for this object
  OptionSet expected_options;
  /// Build method pointer
  BuildPtr build = nullptr;
};

/**
 * The Registry is used as a global singleton to collect information on all available NEML2Object
 * that can be manufactured from the input file.
 *
 * To register a concrete class to the registry, use the macro register_NEML2_object or
 * register_NEML2_object_alias. Each object/class should only be registered once.
 */
class Registry
{
public:
  /// Get the global Registry singleton.
  static Registry & get();

  /// Add information on a NEML2Object to the registry.
  template <typename T>
  static char add(const std::string & name)
  {
    add_inner(name, utils::demangle(typeid(T).name()), T::expected_options(), &build<T>);
    return 0;
  }

  /// Load registry from a dynamic library
  static void load(const std::filesystem::path &);

  /// Check if an object is registered in the registry.
  static bool is_registered(const std::string & name);

  /// Get information of all registered objects.
  static const std::map<std::string, NEML2ObjectInfo> & info();

  /// Get the information of an object given its syntax type
  static const NEML2ObjectInfo & info(const std::string &);

private:
  Registry() = default;

  static void add_inner(const std::string &, const std::string &, const OptionSet &, BuildPtr);

  template <typename T>
  static std::shared_ptr<NEML2Object> build(const OptionSet & options)
  {
    return std::make_shared<T>(options);
  }

  /**
   * @brief Information of all registered objects
   *
   * The key is the name of the object as it appears in the input file, i.e. the right hand side of
   * "type = xxx".
   */
  std::map<std::string, NEML2ObjectInfo> _info;
};
} // namespace neml2
