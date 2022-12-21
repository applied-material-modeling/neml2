#include "neml2/models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
AssociativeIsotropicPlasticHardening::AssociativeIsotropicPlasticHardening(
    const std::string & name, const std::shared_ptr<YieldFunction> & f)
  : PlasticHardening(name),
    yield_function(*f),
    equivalent_plastic_strain_rate(declareOutputVariable<Scalar>(
        {"state", "internal_state", "equivalent_plastic_strain_rate"}))
{
  register_model(f);
  setup();
}

void
AssociativeIsotropicPlasticHardening::set_value(LabeledVector in,
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

  auto gamma_dot = in.get<Scalar>(hardening_rate);
  auto df_dh =
      df_din.get<Scalar>(yield_function.yield_function, yield_function.isotropic_hardening);
  auto ep_dot = -gamma_dot * df_dh;

  out.set(ep_dot, equivalent_plastic_strain_rate);

  if (dout_din)
  {
    // dep_dot/dh = -gamma_dot * d2f/dh2
    auto d2f_dh2 = d2f_din2.get<Scalar>(yield_function.yield_function,
                                        yield_function.isotropic_hardening,
                                        yield_function.isotropic_hardening);

    dout_din->set(-df_dh, equivalent_plastic_strain_rate, hardening_rate);
    dout_din->set(
        -gamma_dot * d2f_dh2, equivalent_plastic_strain_rate, yield_function.isotropic_hardening);
  }
}
} // namespace neml2
