#include "neml2/base/Factory.h"
#include "neml2/drivers/Driver.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto factory = load_input("input.i");
  auto driver = factory->get_driver("driver");
  driver->run();
}
