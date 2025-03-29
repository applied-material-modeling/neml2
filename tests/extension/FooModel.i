[Settings]
  additional_libraries = '/home/gary/projects/neml2/tests/extension/libextension.so'
[]

[Models]
  [foo]
    type = FooModel
    x = 'forces/x'
    y = 'state/y'
    c = 1.1
  []
[]
