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
#include "neml2/csrc/aoti/SimpleScheduler.h"
#include "neml2/csrc/aoti/assertions.h"
#include "neml2/csrc/aoti/factory.h"

namespace neml2::aoti
{
namespace
{
// Map a stub `[Solvers]/<name>` block (carried verbatim from the source input,
// minus `linear_solver`) onto the runtime SolverConfig. Field names + defaults
// mirror the Python Newton / NewtonWithLineSearch HIT schema; line-search knobs
// are honored only for `type = NewtonWithLineSearch` (a plain Newton leaves
// ls_max_iters at the SolverConfig default of 1, i.e. line search disabled).
SolverConfig
parse_solver_config(const nmhit::Node & node)
{
  SolverConfig cfg;
  cfg.atol = node.param_optional<double>("abs_tol", cfg.atol);
  cfg.rtol = node.param_optional<double>("rel_tol", cfg.rtol);
  cfg.miters = static_cast<std::size_t>(
      node.param_optional<std::int64_t>("max_its", static_cast<std::int64_t>(cfg.miters)));

  if (node.param_optional<std::string>("type", std::string{}) == "NewtonWithLineSearch")
  {
    cfg.ls_type = node.param_optional<std::string>("linesearch_type", cfg.ls_type);
    // Python's NewtonWithLineSearch defaults max_linesearch_iterations to 10
    // (which enables line search); not the SolverConfig default of 1.
    cfg.ls_max_iters = static_cast<std::size_t>(
        node.param_optional<std::int64_t>("max_linesearch_iterations", 10));
    cfg.ls_cutback = node.param_optional<double>("linesearch_cutback", cfg.ls_cutback);
    cfg.ls_c = node.param_optional<double>("linesearch_stopping_criteria", cfg.ls_c);
  }
  return cfg;
}
} // namespace

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

  // artifact_path: the per-device artifact folder (absolute per neml2-compile;
  // tolerate a relative path by resolving it against the stub's directory).
  const auto artifact_path_str = model_node->param_optional<std::string>("artifact_path", "");
  _assert(!artifact_path_str.empty(),
          "aoti::load_model: [Models/",
          model_name,
          "] is missing the required 'artifact_path' field.");
  std::filesystem::path artifact_path(artifact_path_str);
  if (artifact_path.is_relative())
    artifact_path = stub_path.parent_path() / artifact_path;

  // Solver config from the referenced [Solvers] block (forward-only models carry
  // none -> C++ defaults apply).
  SolverConfig cfg;
  const auto solver_name = model_node->param_optional<std::string>("solver", std::string{});
  if (!solver_name.empty())
  {
    const nmhit::Node * solver_node = root->find("Solvers/" + solver_name);
    _assert(solver_node != nullptr,
            "aoti::load_model: [Models/",
            model_name,
            "] references solver '",
            solver_name,
            "' but no [Solvers/",
            solver_name,
            "] block exists in '",
            stub_path.string(),
            "'.");
    cfg = parse_solver_config(*solver_node);
  }

  // No scheduler -> pass-through on the default (cpu) device.
  auto sched = scheduler
                   ? std::move(scheduler)
                   : std::static_pointer_cast<WorkScheduler>(
                         std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", 0}));

  // The artifact-root ctor resolves <artifact_path>/<device-type>/*_meta.json
  // for the scheduler's device (and errors if that subfolder is absent).
  DispatchedModel model(artifact_path, std::move(sched));
  model.set_solver_config(cfg);
  return model;
}
} // namespace neml2::aoti
