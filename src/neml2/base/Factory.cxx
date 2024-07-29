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
#include "neml2/base/Registry.h"
#include "neml2/base/HITParser.h"
#include "neml2/base/Settings.h"
#include "neml2/models/Model.h"
#include "neml2/drivers/Driver.h"

namespace neml2
{
void
load_input(const std::filesystem::path & path, const std::string & additional_input)
{
  OptionCollection oc;

  // For now we only support HIT
  if (utils::end_with(path.string(), ".i"))
  {
    HITParser parser;
    oc = parser.parse(path, additional_input);
  }
  else
    throw NEMLException("Unsupported parser type");

  Factory::load_options(oc);
}

void
reload_input(const std::filesystem::path & path, const std::string & additional_input)
{
  Factory::clear();
  load_input(path, additional_input);
}

Model &
get_model(const std::string & mname, bool enable_ad, bool force_create)
{
  OptionSet extra_opts;
  extra_opts.set<bool>("_enable_AD") = enable_ad;
  auto & model = Factory::get_object<Model>("Models", mname, extra_opts, force_create);
  model.reinit();
  return model;
}

Model &
load_model(const std::filesystem::path & path, const std::string & mname, bool enable_ad)
{
  load_input(path);
  return get_model(mname, enable_ad);
}

Model &
reload_model(const std::filesystem::path & path, const std::string & mname, bool enable_ad)
{
  Factory::clear();
  return load_model(path, mname, enable_ad);
}

Driver &
get_driver(const std::string & dname)
{
  return Factory::get_object<Driver>("Drivers", dname);
}

Factory &
Factory::get()
{
  static Factory factory_singleton;
  return factory_singleton;
}

void
Factory::load_options(const OptionCollection & all_options)
{
  auto & factory = get();

  factory._all_options = all_options;

  // Also apply global settings
  factory._settings = Settings(all_options.settings());
}

void
Factory::create_object(const std::string & section, const OptionSet & options)
{
  const std::string & name = options.name();
  const std::string & type = options.type();

  auto builder = Registry::builder(type);
  auto object = (*builder)(options);
  _objects[section][name].push_back(object);
  object->setup();
}

// LCOV_EXCL_START
void
Factory::print(std::ostream & os)
{
  const auto & all_objects = get()._objects;
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
  get()._objects.clear();
}
} // namespace neml2
