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

#include "neml2/drivers/ModelDriver.h"
#include "neml2/misc/assertions.h"
#include "neml2/models/Model.h"

#ifdef NEML2_HAS_DISPATCHER
#include "neml2/dispatchers/valuemap_helpers.h"
#endif

namespace neml2
{
OptionSet
ModelDriver::expected_options()
{
  OptionSet options = Driver::expected_options();

  options.set<std::string>("model");
  options.set("model").doc() = "The material model to be updated by the driver";

  options.set<std::vector<VariableName>>("var_with_intmd_dims") = {};
  options.set("var_with_intmd_dims").doc() =
      "A list of input variable names for which to set intermediate shapes";

  options.set<std::vector<TensorShape>>("var_intmd_shapes") = {};
  options.set("var_intmd_shapes").doc() =
      "A list of tensor shapes corresponding to the variables in 'var_with_intmd_dims'";

  options.set<std::string>("postprocessor");
  options.set("postprocessor").doc() = "The postprocessor model to be applied on the model output";

  options.set<std::string>("device") = "cpu";
  options.set("device").doc() =
      "Device on which to evaluate the material model. The string supplied must follow the "
      "following schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies the device type, "
      "and :<device-index> optionally specifies a device index. For example, device='cpu' sets the "
      "target compute device to be CPU, and device='cuda:1' sets the target compute device to be "
      "CUDA with device ID 1.";

  options.set<bool>("show_model_info") = false;
  options.set("show_model_info").doc() = "Whether to show model information at the beginning";

#ifdef NEML2_HAS_DISPATCHER
  options.set<std::string>("scheduler");
  options.set("scheduler").doc() = "The work scheduler to use";
  options.set<bool>("async_dispatch") = true;
  options.set("async_dispatch").doc() = "Whether to dispatch work asynchronously";
#endif

  return options;
}

ModelDriver::ModelDriver(const OptionSet & options)
  : Driver(options),
    _model(get_model("model")),
    _intmd_vars(options.get<std::vector<VariableName>>("var_with_intmd_dims")),
    _intmd_shapes(options.get<std::vector<TensorShape>>("var_intmd_shapes")),
    _postprocessor(options.get("postprocessor").user_specified() ? get_model("postprocessor")
                                                                 : nullptr),
    _device(options.get<std::string>("device")),
    _show_model_info(options.get<bool>("show_model_info"))
#ifdef NEML2_HAS_DISPATCHER
    ,
    _scheduler(options.get("scheduler").user_specified() ? get_scheduler("scheduler") : nullptr),
    _async_dispatch(options.get<bool>("async_dispatch"))
#endif
{
}

void
ModelDriver::setup()
{
  Driver::setup();

  // Tag intermediate shapes
  for (size_t i = 0; i < _intmd_vars.size(); ++i)
    _model->set_input_intmd_sizes(_intmd_vars[i], _intmd_shapes[i]);

  // Send to device
  _model->to(_device);

#ifdef NEML2_HAS_DISPATCHER
  if (_scheduler)
  {
    auto red = [](std::vector<ValueMap> && results) -> ValueMap
    { return valuemap_cat_reduce(std::move(results), 0); };

    auto post = [this](ValueMap && x) -> ValueMap
    { return valuemap_move_device(std::move(x), _device); };

    auto thread_init = [this](Device device) -> void
    {
      auto new_factory = Factory(_model->factory()->input_file());
      auto new_model = new_factory.get_model(_model->name());
      for (size_t i = 0; i < _intmd_vars.size(); ++i)
        new_model->set_input_intmd_sizes(_intmd_vars[i], _intmd_shapes[i]);
      new_model->to(device);
      _models[std::this_thread::get_id()] = std::move(new_model);
    };

    _dispatcher = std::make_unique<DispatcherType>(
        *_scheduler,
        _async_dispatch,
        [&](ValueMap && x, Device device) -> ValueMap
        {
          auto & model = _async_dispatch ? _models[std::this_thread::get_id()] : _model;

          // If this is not an async dispatch, we need to move the model to the target device
          // _every_ time before evaluation
          if (!_async_dispatch)
            model->to(device);

          neml_assert_dbg(model->variable_options().device() == device);
          return model->value(x);
        },
        red,
        &valuemap_move_device,
        post,
        _async_dispatch ? thread_init : std::function<void(Device)>());
  }
#endif

  // LCOV_EXCL_START
  if (_show_model_info)
    std::cout << *_model << std::endl;
  // LCOV_EXCL_STOP
}

void
ModelDriver::diagnose() const
{
  Driver::diagnose();
  neml2::diagnose(*_model);

  diagnostic_assert(
      _intmd_vars.size() == _intmd_shapes.size(),
      "The number of intermediate variables and intermediate shapes must be the same.");

  if (_postprocessor)
    neml2::diagnose(*_postprocessor);
}
} // namespace neml2
