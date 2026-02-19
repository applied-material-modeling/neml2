#include "neml2/neml2.h"
#include <iostream>

int
main()
{
  neml2::force_link_runtime();
  std::cout << "Number of registered objects: " << neml2::Registry::info().size() << std::endl;
  return 0;
}
