#include "StructuralDriver.h"

using namespace neml2;

StructuralDriver::StructuralDriver(const neml2::Model & model,
                                   torch::Tensor time,
                                   torch::Tensor driving_force,
                                   std::string driving_force_name)
  : _model(model),
    _time(time),
    _driving_force(driving_force),
    _driving_force_name(driving_force_name),
    _nsteps(driving_force.sizes()[0]),
    _nbatch(driving_force.sizes()[1])
{
  neml_assert(time.dim() == 3,
              "time should have dimension 3 "
              "but instead has dimension ",
              time.dim());
  neml_assert(driving_force.dim() == 3,
              driving_force_name,
              " should have dimension 3 "
              "but instead has dimension",
              driving_force.dim());
  neml_assert(time.sizes()[0] == driving_force.sizes()[0],
              driving_force_name,
              " and time should have the "
              "same number of time steps. "
              "The input time has ",
              time.sizes()[0],
              " time steps, "
              "while the input ",
              driving_force_name,
              " has ",
              driving_force.sizes()[0],
              " time steps");
  neml_assert(time.sizes()[1] == driving_force.sizes()[1],
              driving_force_name,
              " and time should have the "
              "same batch size.  The input time has a batch size of ",
              time.sizes()[1],
              " while the input ",
              driving_force_name,
              " has a batch "
              "size of ",
              driving_force.sizes()[1]);
  neml_assert(time.sizes()[2] == 1,
              "Input time should have final "
              "dimension 1 but instead has final dimension ",
              time.sizes()[2]);
  neml_assert(driving_force.sizes()[2] == 6,
              "Input ",
              driving_force_name,
              " should have final "
              "dimension 6 but instead has final dimension ",
              driving_force.sizes()[2]);
}

std::tuple<std::vector<LabeledVector>, std::vector<LabeledVector>>
StructuralDriver::run()
{
  // Create 2 LabeledMatrix to store the inputs and outputs
  std::vector<LabeledVector> all_inputs(_nsteps);
  std::vector<LabeledVector> all_outputs(_nsteps);

  // Initialize
  auto in = LabeledVector(_nbatch, _model.input());
  auto out = LabeledVector(_nbatch, _model.output());

  // Initialize the old state it if necessary
  // For example _model.init_state(in);

  // Initialize the old forces
  Scalar current_time = Scalar(_time.index({0}));
  SymR2 current_driving_force = SymR2(_driving_force.index({0}));
  LabeledVector(in.slice(0, "old_forces")).fill(in.slice(0, "forces"));

  all_inputs[0] = in.clone();
  all_outputs[0] = out.clone();

  for (TorchSize i = 1; i < _nsteps; i++)
  {
    // Advance the step
    current_time = Scalar(_time.index({i}));
    current_driving_force = SymR2(_driving_force.index({i}));
    in.slice(0, "forces").set(current_driving_force, _driving_force_name);
    in.slice(0, "forces").set(current_time, "time");

    // Perform the constitutive update
    out = solve_step(in, i, all_inputs, all_outputs);

    // Propagate the forces and state in time
    // current --> old
    LabeledVector(in.slice(0, "old_state")).fill(out.slice(0, "state"));
    LabeledVector(in.slice(0, "old_forces")).fill(in.slice(0, "forces"));
  }

  return {all_inputs, all_outputs};
}

LabeledVector
StructuralDriver::solve_step(LabeledVector in,
                             TorchSize i,
                             std::vector<LabeledVector> & all_inputs,
                             std::vector<LabeledVector> & all_outputs) const
{
  auto out = _model.value(in);

  // Store the results
  all_inputs[i] = in.clone();
  all_outputs[i] = out.clone();

  return out;
}
