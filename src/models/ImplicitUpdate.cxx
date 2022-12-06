#include "models/ImplicitUpdate.h"

ImplicitUpdate::ImplicitUpdate(const std::string & name,
                               ImplicitModel & model,
                               NonlinearSolver & solver)
  : Model(name),
    _model(registerModel<ImplicitModel>(model)),
    _solver(solver)
{
  // Now that the implicit model has been registered, the input of this ImplicitUpdate model should
  // be the same as the implicit model's input. The input subaxes of the implicit model looks
  // something like
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // inputs: state (this is actually the trial state)
  //         old state
  //         forces
  //         old forces
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // However, the inputs and outputs of *this* model (ImplicitUpdate) should look like this after
  // the update:
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // inputs: old state  ------> outputs: state
  //         forces
  //         old forces
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // as we have eliminated the trial state by solving the nonlinear system.
  // So, we need to remove the "state" subaxis from the input
  output().add<LabeledAxis>("state");
  output().subaxis("state").merge(model.input().subaxis("state"));
  input().remove("state");
  setup();
}

void
ImplicitUpdate::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  TorchSize nbatch = in.batch_size();

  // Cache the input as we are solving the implicit model with FIXED forces, old forces, and old
  // state
  _model.cache_input(in);

  // Solve for the next state
  ImplicitModel::stage = ImplicitModel::Stage::SOLVING;
  BatchTensor<1> sol = _solver.solve(_model, _model.initial_guess(in));
  ImplicitModel::stage = ImplicitModel::Stage::UPDATING;

  out.set(sol, "state");

  // Use the implicit function theorem to calculate the other derivatives
  if (dout_din)
  {
    LabeledVector implicit_in(nbatch, _model.input());
    implicit_in.assemble(in);
    implicit_in.set(sol, "state");

    auto partials = _model.dvalue(implicit_in);
    LabeledMatrix J = partials.slice(1, "state");
    LabeledMatrix Jinv = J.inverse();
    dout_din->block("state", "old_state").copy(-Jinv.chain(partials.slice(1, "old_state")));
    dout_din->block("state", "forces").copy(-Jinv.chain(partials.slice(1, "forces")));
    dout_din->block("state", "old_forces").copy(-Jinv.chain(partials.slice(1, "old_forces")));
  }
}
