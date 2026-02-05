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

#include "neml2/base/OptionSet.h"
#include "neml2/base/Factory.h"

// Registry.h is included here because it is needed for the factory pattern, i.e., for the
// register_NEML2_object macros
#include "neml2/base/Registry.h"

namespace neml2
{
class Settings;
class EquationSystem;
class Solver;
class Data;
class Model;
class Driver;
class WorkScheduler;

/**
 * @brief The base class of all "manufacturable" objects in the NEML2 library.
 *
 * NEML2 uses the standard registry-factory pattern for automatic object registration and creation.
 * The registry and the factory relies on polymophism to collect and resolve all types at run-time.
 * See `Registry` and `Factory` for more details.
 */
class NEML2Object
{
public:
  static OptionSet expected_options();

  NEML2Object() = delete;
  NEML2Object(NEML2Object &&) = delete;
  NEML2Object(const NEML2Object &) = delete;
  NEML2Object & operator=(NEML2Object &&) = delete;
  NEML2Object & operator=(const NEML2Object &) = delete;
  virtual ~NEML2Object() = default;

  /**
   * @brief Construct a new NEML2Object object
   *
   * @param options The set of options extracted from the input file
   */
  NEML2Object(const OptionSet & options);

  const OptionSet & input_options() const { return _input_options; }

  /**
   * @brief Setup this object.
   *
   * This method is called automatically if you use the Factory method get_object or get_object,
   * right after construction. This serves as the entry point for things that are not
   * convenient/possible to do at construction time, but are necessary before this object can be
   * used (by others).
   *
   */
  virtual void setup() {}

  /// A readonly reference to the object's name
  const std::string & name() const { return _input_options.name(); }
  /// A readonly reference to the object's type
  const std::string & type() const { return _input_options.type(); }
  /// A readonly reference to the object's path
  const std::string & path() const { return _input_options.path(); }
  /// A readonly reference to the object's docstring
  const std::string & doc() const { return _input_options.doc(); }

  /// Get the factory that created this object
  Factory * factory() const { return _factory; }

  /// Settings
  const Settings & settings() const { return *_settings; }

  /// Get a readonly pointer to the host
  template <typename T = NEML2Object>
  const T * host() const;

  /// Get a writable pointer to the host
  template <typename T = NEML2Object>
  T * host();

  /// Resolve a TensorName to a Tensor
  template <typename T>
  const T & resolve_tensor(const std::string & name);

  ///@{
  /// Get an object from the factory
  template <class T>
  std::shared_ptr<T> get_object(const std::string & section, const std::string & name);
  /// Get an equation system from the factory
  template <class T = EquationSystem>
  std::shared_ptr<T> get_es(const std::string & name);
  /// Get a solver from the factory
  template <class T = Solver>
  std::shared_ptr<T> get_solver(const std::string & name);
  /// Get a data from the factory
  template <class T = Data>
  std::shared_ptr<T> get_data(const std::string & name);
  /// Get a model from the factory
  template <class T = Model>
  std::shared_ptr<T> get_model(const std::string & name);
  /// Get a driver from the factory
  template <class T = Driver>
  std::shared_ptr<T> get_driver(const std::string & name);
  /// Get a scheduler from the factory
  template <class T = WorkScheduler>
  std::shared_ptr<T> get_scheduler(const std::string & name);
  ///@}

private:
  const OptionSet _input_options;

  /**
   * @brief The factory that created this object
   *
   * @warning This is a pointer to the factory that created this object. Its lifetime is not tied
   * to this object. No guarantees are made that the factory will outlive this object.
   */
  Factory * _factory;

  /// Global settings
  const std::shared_ptr<Settings> _settings;

  /// The publicly exposed NEML2Object
  NEML2Object * _host;
};

template <typename T>
const T *
NEML2Object::host() const
{
  auto host_ptr = dynamic_cast<const T *>(_host ? _host : this);
  if (!host_ptr)
    throw NEMLException("Internal error: Failed to retrieve host of object " + name());
  return host_ptr;
}

template <typename T>
T *
NEML2Object::host()
{
  auto host_ptr = dynamic_cast<T *>(_host ? _host : this);
  if (!host_ptr)
    throw NEMLException("Internal error: Failed to retrieve host of object " + name());
  return host_ptr;
}

template <class T>
std::shared_ptr<T>
NEML2Object::get_object(const std::string & section, const std::string & name)
{
  auto obj_name = _input_options.contains(name) ? _input_options.get<std::string>(name) : name;

  if (!_factory)
    throw NEMLException("Internal error: factory is nullptr for object " + this->name());

  if (!_factory->has_object(section, obj_name))
  {
    if (_input_options.contains(name))
      throw NEMLException(
          path() + " failed to get an object via option '" + name + "' under section " + section +
          ". Currently, " + path() + "/" + name + " = '" + obj_name +
          "'. Check to make sure the object name is specified correctly in the input file.");
    else
      throw NEMLException(path() + " failed to get an object named '" + obj_name +
                          "' under section " + section + ".");
  }

  OptionSet extra_opts;
  extra_opts.set<NEML2Object *>("_host") = host();
  return _factory->get_object<T>(section, obj_name, extra_opts, /*force_create=*/false);
}

template <class T>
std::shared_ptr<T>
NEML2Object::get_es(const std::string & name)
{
  return get_object<T>("EquationSystems", name);
}

template <class T>
std::shared_ptr<T>
NEML2Object::get_solver(const std::string & name)
{
  return get_object<T>("Solvers", name);
}

template <class T>
std::shared_ptr<T>
NEML2Object::get_data(const std::string & name)
{
  return get_object<T>("Data", name);
}

template <class T>
std::shared_ptr<T>
NEML2Object::get_model(const std::string & name)
{
  return get_object<T>("Models", name);
}

template <class T>
std::shared_ptr<T>
NEML2Object::get_driver(const std::string & name)
{
  return get_object<T>("Drivers", name);
}

template <class T>
std::shared_ptr<T>
NEML2Object::get_scheduler(const std::string & name)
{
  return get_object<T>("Schedulers", name);
}
} // namespace neml2
