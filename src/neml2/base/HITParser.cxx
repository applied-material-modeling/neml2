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

#include "neml2/base/HITParser.h"

#include "nmhit/nmhit.h"

#include "neml2/base/Registry.h"
#include "neml2/base/Factory.h"
#include "neml2/base/TensorName.h"
#include "neml2/base/Settings.h"
#include "neml2/base/EnumSelection.h"
#include "neml2/base/MultiEnumSelection.h"
#include "neml2/tensors/tensors.h"
#include "neml2/misc/assertions.h"
#include "neml2/misc/types.h"

namespace neml2
{
namespace
{
/**
 * This class is used to register parsers for custom types with the nmhit library. It is
 * instantiated as a static variable, so its constructor is called before main() is executed,
 * ensuring that the parsers are registered before any parsing is attempted.
 */
struct NMHITRegistrar
{
  NMHITRegistrar()
  {
    nmhit::TypeRegistry::register_parser<TensorShape>([](const std::string & s)
                                                      { return utils::parse<TensorShape>(s); });
    nmhit::TypeRegistry::register_parser<Device>([](const std::string & s)
                                                 { return utils::parse<Device>(s); });

#define register_tensor_name(T)                                                                    \
  nmhit::TypeRegistry::register_parser<TensorName<T>>([](const std::string & s)                    \
                                                      { return TensorName<T>(s); });               \
  static_assert(true)
    FOR_ALL_TENSORBASE(register_tensor_name);
    register_tensor_name(ATensor);
#undef register_tensor_name
  }
};
static NMHITRegistrar _nmhit_registrar;
} // namespace

// Forward declarations
static InputFile parse_hit(const nmhit::Node * root);
static void extract_options(const nmhit::Node * object, OptionSet & options);
static void extract_option(const nmhit::Node * node, OptionSet & options);

InputFile
HITParser::parse(const std::filesystem::path & filename, const std::string & additional_input) const
{
  // Parse the file; additional_input is appended as a post-snippet (use := for overrides)
  auto root = nmhit::parse_file(filename, {}, {additional_input});
  return parse_hit(root.get());
}

InputFile
parse_hit(const nmhit::Node * root)
{
  // Extract global settings
  OptionSet settings = Settings::expected_options();
  if (auto * const node = root->find("Settings"))
    extract_options(node, settings);

  // Loop over each known section and extract options for each object
  InputFile inp(settings);
  for (const auto & section : Parser::sections)
  {
    auto * section_node = root->find(section);
    if (section_node)
    {
      auto objects = section_node->children(nmhit::NodeType::Section);
      for (auto * object : objects)
      {
        std::string type = object->param<std::string>("type");
        const auto * info = Registry::info(type);
        if (!info)
        {
          std::cerr << "Warning: Object of type '" << type
                    << "' is not registered in the NEML2 registry. This object will be ignored.\n";
          continue;
        }

        // Fill in the metadata
        auto options = info->expected_options;
        options.name() = object->path();
        options.type() = type;
        options.path() = object->fullpath();

        extract_options(object, options);
        inp[section][options.name()] = options;
      }
    }
  }

  return inp;
}

void
extract_options(const nmhit::Node * object, OptionSet & options)
{
  for (auto * node : object->children(nmhit::NodeType::Field))
    if (node->path() != "type")
      extract_option(node, options);

  // check if all required options are defined
  std::stringstream ss;
  for (const auto & [name, option] : options)
    if (option->required() && !option->defined())
      ss << options.path() << ": Option '" << option->name()
         << "' is required but not specified. Description: " << option->doc() << "\n";
  neml_assert(ss.str().empty(), ss.str());
}

// NOLINTBEGIN
void
extract_option(const nmhit::Node * n, OptionSet & options)
{
#define try_param(ptype)                                                                           \
  else if (tp ==                                                                                   \
           utils::demangle(typeid(ptype).name())) dynamic_cast<Option<ptype> *>(option.get())      \
      ->set() = n->param<ptype>()

#define try_param_t(ptype)                                                                         \
  try_param(ptype);                                                                                \
  try_param(std::vector<ptype>);                                                                   \
  try_param(std::vector<std::vector<ptype>>)

#define try_tensor_name(ptype) try_param_t(TensorName<ptype>)

  if (n->type() != nmhit::NodeType::Field)
    return;

  bool found = false;
  for (auto & [name, option] : options)
    if (name == n->path())
    {
      neml_assert(!option->suppressed(),
                  "Option named '",
                  option->name(),
                  "' is suppressed, and its value cannot be modified.");

      found = true;
      const auto & tp = option->type();

      if (false)
        ;
      try_param_t(TensorShape);
      try_param_t(bool);
      try_param_t(int);
      try_param_t(unsigned int);
      try_param_t(std::size_t);
      try_param_t(Size);
      try_param_t(double);
      try_param_t(std::string);
      try_param_t(Device);
      FOR_ALL_TENSORBASE(try_tensor_name);
      try_tensor_name(ATensor);
      else if (tp == utils::demangle(typeid(EnumSelection).name()))
      {
        auto * option_enum = dynamic_cast<Option<EnumSelection> *>(option.get());
        neml_assert(
            option_enum, "Option named '", option->name(), "' is not of type EnumSelection.");
        option_enum->set().select(n->param<std::string>());
      }
      else if (tp == utils::demangle(typeid(MultiEnumSelection).name()))
      {
        auto * option_multi_enum = dynamic_cast<Option<MultiEnumSelection> *>(option.get());
        neml_assert(option_multi_enum,
                    "Option named '",
                    option->name(),
                    "' is not of type MultiEnumSelection.");
        option_multi_enum->set().select(n->param<std::vector<std::string>>());
      }
      // LCOV_EXCL_START
      else neml_assert(false, "Unsupported option type for option ", n->fullpath());
      // LCOV_EXCL_STOP

      option->user_specified() = true;
      option->defined() = true;

      break;
    }
  neml_assert(found, "Unused option ", n->fullpath());
}
// NOLINTEND

} // namespace neml2
