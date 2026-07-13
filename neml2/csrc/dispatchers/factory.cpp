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

#include <cstdint>
#include <utility>

#include <nmhit/nmhit.h>

#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/dispatchers/SimpleScheduler.h"
#include "neml2/csrc/aoti/assertions.h"
#include "neml2/csrc/dispatchers/factory.h"

namespace neml2::aoti
{
DispatchedModel
load_model(const std::filesystem::path & stub_path,
           const std::string & model_name,
           std::shared_ptr<WorkScheduler> scheduler)
{
  std::unique_ptr<nmhit::Node> root;
  try
  {
    root = nmhit::parse_file(stub_path);
  }
  catch (const std::exception & e)
  {
    _throw("aoti::load_model: failed to parse stub '", stub_path.string(), "': ", e.what());
  }

  // [Models]/<model_name> -- must be an AOTIModel shim.
  const nmhit::Node * model_node = root->find("Models/" + model_name);
  _assert(model_node != nullptr,
          "aoti::load_model: no [Models/",
          model_name,
          "] block in '",
          stub_path.string(),
          "'.");
  const auto type = model_node->param_optional<std::string>("type", std::string{});
  _assert(type == "AOTIModel",
          "aoti::load_model: [Models/",
          model_name,
          "] has type '",
          type,
          "', expected 'AOTIModel'. Pass the stub produced by `neml2-compile`.");

  // artifact_path: the artifact root directory. neml2-compile writes it relative
  // to the stub (so the stub + folder relocate together as a portable bundle);
  // resolve a relative value against the stub's directory. Absolute is honored too.
  const auto artifact_path_str = model_node->param_optional<std::string>("artifact_path", "");
  _assert(!artifact_path_str.empty(),
          "aoti::load_model: [Models/",
          model_name,
          "] is missing the required 'artifact_path' field.");
  std::filesystem::path artifact_path(artifact_path_str);
  if (artifact_path.is_relative())
    artifact_path = stub_path.parent_path() / artifact_path;

  // No scheduler -> pass-through on the default (cpu) device.
  auto sched = scheduler
                   ? std::move(scheduler)
                   : std::static_pointer_cast<WorkScheduler>(
                         std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", 0}));

  // The artifact-root ctor resolves the shared <artifact_path>/metadata.json and
  // the per-<device>/<dtype>/ binaries for the scheduler's device (erroring if the
  // leaf is absent). Solver config is read from the shared metadata by each
  // per-device Model. `set_solver_config` remains available to a host that wants
  // to override it at runtime.
  return DispatchedModel(artifact_path, std::move(sched));
}
} // namespace neml2::aoti
