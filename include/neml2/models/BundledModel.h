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

#ifdef NEML2_HAS_ZLIB
#include <zlib.h>
#endif

#ifdef NEML2_HAS_JSON
#include "nlohmann/json.hpp"
#endif

#define NEML2_CAN_BUNDLE_MODEL defined(NEML2_HAS_ZLIB) && defined(NEML2_HAS_JSON)

#include "neml2/models/Model.h"

namespace neml2
{
#ifdef NEML2_CAN_BUNDLE_MODEL
void bundle_model(const std::string & file,
                  const std::string & name,
                  const std::string & cliargs = "",
                  const nlohmann::json & config = nlohmann::json(),
                  std::filesystem::path output_path = std::filesystem::path());

std::pair<std::shared_ptr<Model>, nlohmann::json> unbundle_model(const std::filesystem::path & pkg,
                                                                 NEML2Object * host = nullptr);
#endif // NEML2_CAN_BUNDLE_MODEL

class BundledModel : public Model
{
public:
  static OptionSet expected_options();

  BundledModel(const OptionSet & options);

  const nlohmann::json & config() const { return _config; }

  ///@{
  /// Methods for retrieving descriptions
  std::string description() const;
  std::string input_description(const VariableName & name) const;
  std::string output_description(const VariableName & name) const;
  std::string param_description(const std::string & name) const;
  std::string buffer_description(const std::string & name) const;
  ///@}

protected:
  void link_output_variables() override;
  void set_value(bool, bool, bool) override;

  std::shared_ptr<Model> _archived_model;

  nlohmann::json _config;
};
} // namespace neml2
