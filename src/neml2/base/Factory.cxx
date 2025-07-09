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

#include "neml2/base/Factory.h"
#include "neml2/base/InputFile.h"
#include "neml2/base/Settings.h"
#include "neml2/base/OptionSet.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/base/Registry.h"
#include "neml2/base/HITParser.h"

namespace neml2
{
std::unique_ptr<Factory>
load_input(const std::filesystem::path & path, const std::string & additional_input)
{
  // For now we only support HIT
  if (utils::end_with(path.string(), ".i"))
  {
    HITParser parser;
    auto inp = parser.parse(path, additional_input);
    return std::make_unique<Factory>(inp);
  }
  else
    throw ParserException("Unsupported parser type");
}

Factory::Factory(InputFile inp)
  : _input_file(std::move(inp)),
    _settings(std::make_shared<Settings>(_input_file.settings())),
    _objects()
{
}

bool
Factory::has_object(const std::string & section, const std::string & name)
{
  return _objects.count(section) && _objects.at(section).count(name);
}

void
Factory::create_object(const std::string & section, const OptionSet & options)
{
  const std::string & name = options.name();
  const std::string & type = options.type();

  auto build = Registry::info(type).build;
  auto object = (*build)(options);
  _objects[section][name].push_back(object);

  try
  {
    object->setup();
  }
  catch (const std::exception & e)
  {
    throw FactoryException("Failed to setup object '" + name + "' of type '" + type +
                           "' in section '" + section + "':\n" + e.what());
  }

  // Record the object options in the serialized file if we are serializing
  if (_serializing)
    (*_serialized_file)[section][name] = object->input_options();
}

std::unique_ptr<InputFile>
Factory::serialize_object(const std::string & section,
                          const std::string & name,
                          const OptionSet & additional_options)
{
  _serialized_file = std::make_unique<InputFile>(_input_file.settings());
  _serializing = true;
  get_object<NEML2Object>(section, name, additional_options);
  _serializing = false;
  return std::move(_serialized_file);
}

bool
Factory::options_compatible(const std::shared_ptr<NEML2Object> & obj, const OptionSet & opts) const
{
  return neml2::options_compatible(obj->input_options(), opts);
}

// LCOV_EXCL_START
void
Factory::print(std::ostream & os)
{
  const auto & all_objects = _objects;
  for (const auto & [section, objects] : all_objects)
  {
    os << "- " << section << ":" << std::endl;
    for (const auto & object : objects)
      os << "   " << object.first << ": " << object.second.size() << std::endl;
  }
}
// LCOV_EXCL_STOP

void
Factory::clear()
{
  _objects.clear();
}
} // namespace neml2
