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

#include <filesystem>
#include <iostream>

#include "neml2/misc/errors.h"
#include "neml2/base/Settings.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/base/InputFile.h"

namespace neml2
{
// Forward decl
class Settings;
class Factory;

/**
 * @brief A convenient function to parse all options from an input file
 *
 * Previously loaded input options will be discarded!
 *
 * @warning All threads share the same input options, so in principle this function is not intended
 * to be called inside a threaded region.
 *
 * @param path Path to the input file to be parsed
 * @param additional_input Additional cliargs to pass to the parser
 */
Factory load_input(const std::filesystem::path & path, const std::string & additional_input = "");

/**
 * The factory is responsible for:
 * 1. retriving a NEML2Object given the object name as a std::string
 * 2. creating a NEML2Object given the type of the NEML2Object as a std::string.
 */
class Factory
{
public:
  Factory(const InputFile &);

  /// Get the input file
  InputFile & input_file() { return _input_file; }

  /// Get the input file
  const InputFile & input_file() const { return _input_file; }

  /// Global settings
  const std::shared_ptr<Settings> & settings() const { return _input_file.settings(); }

  /// Check if an object with the given name exists under the given section.
  bool has_object(const std::string & section, const std::string & name);

  /**
   * @brief Retrive an object pointer under the given section with the given object name.
   *
   * An exception is thrown if
   * - the object with the given name does not exist, or
   * - the object with the given name exists but does not have the correct type (e.g., dynamic case
   * fails).
   *
   * @tparam T The type of the NEML2Object
   * @param section The section name under which the search happens.
   * @param name The name of the object to retrieve.
   * @param additional_options Additional input options to pass to the object constructor
   * @param force_create (Optional) Force the factory to create a new object even if the object has
   * already been created.
   * @return std::shared_ptr<T> The object pointer.
   */
  template <class T>
  std::shared_ptr<T> get_object(const std::string & section,
                                const std::string & name,
                                const OptionSet & additional_options = OptionSet(),
                                bool force_create = true);

  /// @brief Delete all factories and destruct all the objects.
  void clear();

  /**
   * @brief List all the manufactured objects.
   *
   * @param os The stream to write to.
   */
  void print(std::ostream & os = std::cout);

protected:
  /**
   * @brief Manufacture a single NEML2Object.
   *
   * @param section The section which the object to be manufactured belongs to.
   * @param options The options of the object.
   */
  void create_object(const std::string & section, const OptionSet & options);

private:
  /// The input file
  InputFile _input_file;

  /**
   * Manufactured objects. The key of the outer map is the section name, and the key of the inner
   * map is the object name.
   */
  std::map<std::string, std::map<std::string, std::vector<std::shared_ptr<NEML2Object>>>> _objects;
};

template <class T>
inline std::shared_ptr<T>
Factory::get_object(const std::string & section,
                    const std::string & name,
                    const OptionSet & additional_options,
                    bool force_create)
{
  if (input_file().data().empty())
    throw FactoryException("The input file is empty.");

  // Easy if it already exists
  if (!force_create)
    if (_objects.count(section) && _objects.at(section).count(name))
      for (const auto & neml2_obj : _objects[section][name])
      {
        // Check for option clash
        if (!options_compatible(neml2_obj->input_options(), additional_options))
          continue;

        // Check for object type
        auto obj = std::dynamic_pointer_cast<T>(neml2_obj);
        if (!obj)
          throw FactoryException(
              "Found object named " + name + " under section " + section +
              ". But dynamic cast failed. Did you specify the correct object type?");

        return obj;
      }

  // Otherwise try to create it
  for (const auto & options : _input_file[section])
    if (options.first == name)
    {
      auto new_options = options.second;
      new_options.set<Factory *>("_factory") = this;
      new_options.set<std::shared_ptr<Settings>>("_settings") = settings();
      new_options += additional_options;
      create_object(section, new_options);
      break;
    }

  if (!_objects.count(section) || !_objects.at(section).count(name))
    throw FactoryException("Failed to get object named " + name + " under section " + section +
                           ". Check to make sure the object is defined in the input file.");

  auto obj = std::dynamic_pointer_cast<T>(_objects[section][name].back());

  if (!obj)
    throw FactoryException("Internal error: Factory failed to create object " + name);

  return obj;
}
} // namespace neml2
