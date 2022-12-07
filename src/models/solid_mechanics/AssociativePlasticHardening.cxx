#include "models/solid_mechanics/AssociativePlasticHardening.h"
#include "tensors/SymSymR4.h"

AssociativePlasticHardening::AssociativePlasticHardening(const std::string & name,
                                                         YieldFunction & f)
  : PlasticHardening(name),
    yield_function(registerModel<YieldFunction>(f))
{
  setup();
}

void
AssociativePlasticHardening::set_value(LabeledVector in,
                                       LabeledVector out,
                                       LabeledMatrix * dout_din) const
{
  // For associative flow,
  // ep_dot = - gamma_dot * df/dh
  TorchSize nbatch = in.batch_size();
  LabeledMatrix df_din(nbatch, yield_function.output(), yield_function.input());
  LabeledTensor<1, 3> d2f_din2(
      nbatch, yield_function.output(), yield_function.input(), yield_function.input());

  if (dout_din)
    std::tie(df_din, d2f_din2) = yield_function.dvalue_and_d2value(in);
  else
    df_din = yield_function.dvalue(in);

  auto gamma_dot = in.slice(0, "state").get<Scalar>("hardening_rate");
  auto df_dh = df_din.block("state", "state").get<Scalar>("yield_function", "isotropic_hardening");
  auto ep_dot = -gamma_dot * df_dh;

  out.slice(0, "state").set(ep_dot, "equivalent_plastic_strain_rate");

  if (dout_din)
  {
    // For associative flow,
    // ep_dot = -gamma_dot * df/dh
    // So dep_dot/dh = -gamma_dot * d2f/dh2
    auto d2f_dh2 = d2f_din2.block("state", "state", "state")
                       .get<Scalar>("yield_function", "isotropic_hardening", "isotropic_hardening");

    dout_din->block("state", "state")
        .set(-df_dh, "equivalent_plastic_strain_rate", "hardening_rate");
    dout_din->block("state", "state")
        .set(-gamma_dot * d2f_dh2, "equivalent_plastic_strain_rate", "isotropic_hardening");
  }
}
