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

#include "hit/braceexpr.h"

#include "neml2/base/HITParser.h"
#include "neml2/base/Registry.h"
#include "neml2/base/Factory.h"
#include "neml2/base/TensorName.h"
#include "neml2/base/Settings.h"
#include "neml2/base/EnumSelection.h"
#include "neml2/base/MultiEnumSelection.h"
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/misc/errors.h"
#include "neml2/misc/string_utils.h"
#include "neml2/tensors/tensors.h"
#include "neml2/misc/assertions.h"
#include "neml2/misc/types.h"
#include <hit/parse.h>

namespace neml2
{
InputFile
HITParser::parse(const std::filesystem::path & filename, const std::string & additional_input) const
{
  // Open and read the file
  std::ifstream file(filename);
  neml_assert(file.is_open(), "Unable to open file ", filename);

  // Read the file into a string
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string input = buffer.str();

  // Let HIT lex the string
  std::unique_ptr<hit::Node> root(hit::parse(filename, input));
  neml_assert(root.get(), "HIT failed to lex the input file: ", filename);

  // Handle additional input (they could be coming from cli args)
  std::unique_ptr<hit::Node> cli_root(hit::parse("cliargs", additional_input));
  hit::merge(cli_root.get(), root.get());

  // Preevaluate the input
  hit::BraceExpander expander;
  hit::EnvEvaler env;
  hit::RawEvaler raw;
  expander.registerEvaler("env", env);
  expander.registerEvaler("raw", raw);
  root->walk(&expander);

  return parse_from_hit_node(root.get());
}

InputFile
HITParser::parse_from_hit_node(hit::Node * root) const
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
      auto objects = section_node->children(hit::NodeType::Section);
      for (auto * object : objects)
      {
        auto options = extract_object_options(object, section_node);
        inp[section][options.name()] = options;
      }
    }
  }

  return inp;
}

OptionSet
HITParser::extract_object_options(hit::Node * object, hit::Node * section) const
{
  // There is a special field reserved for object type
  std::string type = object->param<std::string>("type");
  // Extract the options
  auto options = Registry::info(type).expected_options;
  extract_options(object, options);

  // Also fill in the metadata
  options.name() = object->path();
  options.type() = type;
  options.path() = section->fullpath();

  return options;
}

void
HITParser::extract_options(hit::Node * object, OptionSet & options) const
{
  for (auto * node : object->children(hit::NodeType::Field))
    if (node->path() != "type")
      extract_option(node, options);
}

// NOLINTBEGIN
void
HITParser::extract_option(hit::Node * n, OptionSet & options) const
{
#define extract_option_base(ptype, method)                                                         \
  else if (option->type() == utils::demangle(typeid(ptype).name()))                                \
      neml_assert(method(dynamic_cast<Option<ptype> *>(option.get())->set(), n->strVal()),         \
                  utils::parse_failure_message<ptype>(n->strVal()))

#define extract_option_t(ptype)                                                                    \
  extract_option_base(ptype, utils::parse_<ptype>);                                                \
  extract_option_base(std::vector<ptype>, utils::parse_vector_<ptype>);                            \
  extract_option_base(std::vector<std::vector<ptype>>, utils::parse_vector_vector_<ptype>)

#define extract_option_tensor_t(ptype) extract_option_t(TensorName<ptype>)

  if (n->type() == hit::NodeType::Field)
  {
    bool found = false;
    for (auto & [name, option] : options)
      if (name == n->path())
      {
        neml_assert(!option->suppressed(),
                    "Option named '",
                    option->name(),
                    "' is suppressed, and its value cannot be modified.");

        found = true;

        if (false)
          ;
        extract_option_t(TensorShape);
        extract_option_t(bool);
        extract_option_t(int);
        extract_option_t(unsigned int);
        extract_option_t(std::size_t);
        extract_option_t(Size);
        extract_option_t(double);
        extract_option_t(std::string);
        extract_option_t(VariableName);
        extract_option_t(EnumSelection);
        extract_option_t(MultiEnumSelection);
        FOR_ALL_TENSORBASE(extract_option_tensor_t);
        extract_option_tensor_t(ATensor);
        extract_option_t(Device);
        // LCOV_EXCL_START
        else neml_assert(false, "Unsupported option type for option ", n->fullpath());
        // LCOV_EXCL_STOP

        option->user_specified() = true;

        break;
      }
    neml_assert(found, "Unused option ", n->fullpath());
  }
}
// NOLINTEND

std::string
HITParser::serialize(const InputFile & inp) const
{
  // Serialized input
  std::string output;

  // Serialize settings
  auto node = std::make_unique<hit::Section>("Settings");
  serialize_options(node.get(), inp.settings());
  output += node->render();

  // Serialize each section
  for (const auto & [section_name, section] : inp.data())
  {
    output += "\n\n"; // Add a newline before each section for readability
    auto section_node = std::make_unique<hit::Section>(section_name);
    for (const auto & [object_name, options] : section)
    {
      // Create a section for the object
      auto object_node = std::make_unique<hit::Section>(object_name);
      // Add the special field "type". The destructor will delete the children nodes, so it's
      // generally safe to use raw pointers for children
      object_node->addChild(new hit::Field("type", hit::Field::Kind::String, options.type()));
      // Serialize the options
      serialize_options(object_node.get(), options);
      // Add the object node to the section
      section_node->addChild(object_node.release());
    }
    output += section_node->render();
  }

  return output;
}

void
HITParser::serialize_options(hit::Node * node, const OptionSet & options) const
{
  for (const auto & [key, val] : options)
  {
    if (val->suppressed() || !val->user_specified())
      continue;

    // auto * field = serialize_option(key, val.get());
    auto val_str = "'" + utils::stringify(*val) + "'"; // Wrap in single quotes
    auto * field = new hit::Field(key, hit::Field::Kind::String, val_str);
    node->addChild(field);
  }
}
} // namespace neml2
