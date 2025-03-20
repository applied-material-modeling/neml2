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

  options.set<std::string>("device") = "cpu";
  options.set("device").doc() =
      "Device on which to evaluate the material model. The string supplied must follow the "
      "following schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies the device type, "
      "and :<device-index> optionally specifies a device index. For example, device='cpu' sets the "
      "target compute device to be CPU, and device='cuda:1' sets the target compute device to be "
      "CUDA with device ID 1.";

  options.set<bool>("show_parameters") = false;
  options.set("show_parameters").doc() = "Whether to show model parameters at the beginning";
  options.set<bool>("show_input_axis") = false;
  options.set("show_input_axis").doc() = "Whether to show model input axis at the beginning";
  options.set<bool>("show_output_axis") = false;
  options.set("show_output_axis").doc() = "Whether to show model output axis at the beginning";

#ifdef NEML2_HAS_DISPATCHER
  options.set<std::string>("scheduler");
  options.set("scheduler").doc() = "The work scheduler to use";
  options.set<bool>("async_dispatch") = false;
  options.set("async_dispatch").doc() = "Whether to dispatch work asynchronously";
#endif

  return options;
}

ModelDriver::ModelDriver(const OptionSet & options)
  : Driver(options),
    _model(get_model(options.get<std::string>("model"))),
    _device(options.get<std::string>("device")),
    _show_params(options.get<bool>("show_parameters")),
    _show_input(options.get<bool>("show_input_axis")),
    _show_output(options.get<bool>("show_output_axis"))
#ifdef NEML2_HAS_DISPATCHER
    ,
    _scheduler(options.get("scheduler").user_specified()
                   ? Factory::get_object_ptr<WorkScheduler>("Schedulers",
                                                            options.get<std::string>("scheduler"))
                   : nullptr),
    _async_dispatch(options.get<bool>("async_dispatch"))
#endif
{
}

void
ModelDriver::setup()
{
  Driver::setup();

#ifdef NEML2_HAS_DISPATCHER
  if (_scheduler)
  {
    auto red = [](std::vector<ValueMap> && results) -> ValueMap
    { return valuemap_cat_reduce(std::move(results), 0); };

    auto post = [this](ValueMap && x) -> ValueMap
    { return valuemap_move_device(std::move(x), _device); };

    auto thread_init = [this](Device device) -> void
    {
      auto & model = get_model(_model.name());
      model.to(device);
    };

    _dispatcher = std::make_unique<DispatcherType>(
        *_scheduler,
        _async_dispatch,
        [&](ValueMap && x, Device device) -> ValueMap
        {
          auto & model = get_model(_model.name());

          // If this is not an async dispatch, we need to move the model to the target device
          // _every_ time before evaluation
          if (!_async_dispatch)
            model.to(device);

          neml_assert_dbg(model.tensor_options().device() == device);
          return model.value(std::move(x));
        },
        red,
        &valuemap_move_device,
        post,
        _async_dispatch ? thread_init : std::function<void(Device)>());
  }
#endif

  // LCOV_EXCL_START
  if (_show_params)
    for (auto && [pname, pval] : _model.named_parameters())
      std::cout << pname << std::endl;

  if (_show_input)
    std::cout << _model.name() << "'s input axis:\n" << _model.input_axis() << std::endl;

  if (_show_output)
    std::cout << _model.name() << "'s output axis:\n" << _model.output_axis() << std::endl;
  // LCOV_EXCL_STOP
}

void
ModelDriver::diagnose() const
{
  Driver::diagnose();
  neml2::diagnose(_model);
}
} // namespace neml2
