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

#include "neml2/models/BundledModel.h"
#include "neml2/base/HITParser.h"

namespace neml2
{
static std::string
pack_strings(const std::vector<std::string> & strings)
{
  std::string packed;
  for (const auto & s : strings)
  {
    std::uint32_t len = static_cast<std::uint32_t>(s.size());
    packed.append(reinterpret_cast<const char *>(&len), sizeof(len));
    packed.append(s);
  }
  return packed;
}

static std::vector<std::string>
unpack_strings(const std::string & packed)
{
  std::vector<std::string> strings;
  std::size_t offset = 0;
  while (offset + sizeof(std::uint32_t) <= packed.size())
  {
    std::uint32_t len = 0;
    std::memcpy(&len, packed.data() + offset, sizeof(len));
    offset += sizeof(len);
    if (offset + len > packed.size())
      throw NEMLException("Invalid packed string format");
    strings.emplace_back(packed.data() + offset, len);
    offset += len;
  }
  return strings;
}

#ifdef NEML2_CAN_BUNDLE_MODEL
void
bundle_model(const std::string & file,
             const std::string & name,
             const std::string & cliargs,
             const nlohmann::json & config,
             std::filesystem::path output_path)
{
  namespace fs = std::filesystem;

  // load the input file and serialize the model
  auto factory = load_input(file, cliargs);
  auto inp = factory->serialize_object("Models", name);
  auto parser = HITParser(); // For now we always use HIT for serialization
  auto inp_str = parser.serialize(*inp);
  auto model = factory->get_model(name);

  // default output name
  if (output_path.empty())
    output_path =
        fs::path(file).parent_path() / (fs::path(file).stem().string() + "_" + name + ".gz");

  // create the header
  nlohmann::json header;
  header["schema"] = 1;
  header["name"] = name;

  if (config.contains("desc"))
    header["desc"] = config["desc"];

  for (auto && [vname, var] : model->input_variables())
    if (config.contains("inputs") && config["inputs"].contains(vname.str()) &&
        config["inputs"][vname.str()].contains("desc"))
      header["inputs"][vname.str()]["desc"] = config["inputs"][vname.str()]["desc"];

  for (auto && [vname, var] : model->output_variables())
    if (config.contains("outputs") && config["outputs"].contains(vname.str()) &&
        config["outputs"][vname.str()].contains("desc"))
      header["outputs"][vname.str()]["desc"] = config["outputs"][vname.str()]["desc"];

  for (auto && [pname, param] : model->named_parameters())
    if (config.contains("parameters") && config["parameters"].contains(pname) &&
        config["parameters"][pname].contains("desc"))
      header["parameters"][pname]["desc"] = config["parameters"][pname]["desc"];

  for (auto && [bname, buffer] : model->named_buffers())
    if (config.contains("buffers") && config["buffers"].contains(bname) &&
        config["buffers"][bname].contains("desc"))
      header["buffers"][bname]["desc"] = config["buffers"][bname]["desc"];

  // pack strings for serialization
  const auto packed_str = pack_strings({header.dump(), inp_str});

  // stream the serialized model to a file
  gzFile out_file = gzopen(fs::absolute(output_path).c_str(), "wb");
  if (!out_file)
    throw NEMLException("Failed to open output file for writing: " + output_path.string());
  if (gzwrite(out_file, packed_str.c_str(), packed_str.size()) !=
      static_cast<int>(packed_str.size()))
  {
    gzclose(out_file);
    throw NEMLException("Failed to write to output file: " + output_path.string());
  }
  if (gzclose(out_file) != Z_OK)
    throw NEMLException("Failed to close output file: " + output_path.string());
}

std::pair<std::shared_ptr<Model>, nlohmann::json>
unbundle_model(const std::filesystem::path & pkg, NEML2Object * host)
{
  namespace fs = std::filesystem;

  // deserialize from gz
  gzFile in_file = gzopen(fs::absolute(pkg).c_str(), "rb");
  if (!in_file)
    throw NEMLException("Failed to open packaged model file for reading: " + pkg.string());
  std::string packed_str;
  std::array<char, 4096> buffer{};
  int bytes_read = 0;
  while ((bytes_read = gzread(in_file, buffer.data(), buffer.size())) > 0)
    packed_str.append(buffer.data(), bytes_read);
  if (gzclose(in_file) != Z_OK)
    throw NEMLException("Failed to close input file: " + pkg.string());

  // unpack the strings
  auto unpacked = unpack_strings(packed_str);
  if (unpacked.size() != 2)
    throw NEMLException("Invalid packed model format: expected 2 strings, got " +
                        std::to_string(unpacked.size()));
  const auto & header_str = unpacked[0];
  const auto & inp_str = unpacked[1];

  // parse the header
  auto config = nlohmann::json::parse(header_str);

  // parse the input string
  auto parser = HITParser(); // For now we always use HIT for serialization
  auto inp = parser.parse_from_string(inp_str);
  auto factory = std::make_unique<Factory>(inp);
  OptionSet additional_options;
  additional_options.set<NEML2Object *>("host") = host;
  auto model = factory->get_object<Model>("Models", config["name"], additional_options);

  return {std::move(model), config};
}
#endif // NEML2_CAN_BUNDLE_MODEL

register_NEML2_object(BundledModel);

OptionSet
BundledModel::expected_options()
{
  OptionSet options = Model::expected_options();
  NonlinearSystem::enable_automatic_scaling(options);
  options.doc() =
      "Deserialize a model from an archive and use it as a new model. The deserialized model "
      "is a 'black box' and can be used in the same way as any other models.";

  options.set<std::string>("archive");
  options.set("archive").doc() =
      "Path to the archived model file. The file must be a valid archived model file generated by "
      "NEML2. The path can either be relative or absolute. If a relative path is provided, it will "
      "be resolved against the current working directory.";

  return options;
}

BundledModel::BundledModel(const OptionSet & options)
  : Model(options)
{
#ifdef NEML2_CAN_BUNDLE_MODEL
  // Unpack the model
  std::tie(_archived_model, _config) = unbundle_model(options.get<std::string>("archive"), host());

  // register this model so that it travels with this model
  _registered_models.push_back(_archived_model);

  // clone the input and output variables to the current model
  for (auto && [name, var] : _archived_model->input_variables())
    clone_input_variable(*var);
  for (auto && [name, var] : _archived_model->output_variables())
    clone_output_variable(*var);
#else
  throw NEMLException("NEML2 was not built with support for bundled models.");
#endif // NEML2_CAN_BUNDLE_MODEL
}

std::string
BundledModel::description() const
{
  if (_config.contains("desc"))
    return _config["desc"].get<std::string>();
  return "";
}

std::string
BundledModel::input_description(const VariableName & name) const
{
  if (_config.contains("inputs") && _config["inputs"].contains(name.str()) &&
      _config["inputs"][name.str()].contains("desc"))
    return _config["inputs"][name.str()]["desc"].get<std::string>();
  return "";
}

std::string
BundledModel::output_description(const VariableName & name) const
{
  if (_config.contains("outputs") && _config["outputs"].contains(name.str()) &&
      _config["outputs"][name.str()].contains("desc"))
    return _config["outputs"][name.str()]["desc"].get<std::string>();
  return "";
}

std::string
BundledModel::param_description(const std::string & name) const
{
  if (_config.contains("parameters") && _config["parameters"].contains(name) &&
      _config["parameters"][name].contains("desc"))
    return _config["parameters"][name]["desc"].get<std::string>();
  return "";
}

std::string
BundledModel::buffer_description(const std::string & name) const
{
  if (_config.contains("buffers") && _config["buffers"].contains(name) &&
      _config["buffers"][name].contains("desc"))
    return _config["buffers"][name]["desc"].get<std::string>();
  return "";
}

void
BundledModel::link_output_variables()
{
  Model::link_output_variables();
  for (auto && [name, var] : output_variables())
    var->ref(_archived_model->output_variable(name));
}

void
BundledModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
  _archived_model->forward_maybe_jit(out, dout_din, d2out_din2);

  // copy derivatives
  if (dout_din)
    for (auto && [name, var] : output_variables())
      var->derivatives() = _archived_model->output_variable(name).derivatives();
  if (d2out_din2)
    for (auto && [name, var] : output_variables())
      var->second_derivatives() = _archived_model->output_variable(name).second_derivatives();
}
} // namespace neml2
