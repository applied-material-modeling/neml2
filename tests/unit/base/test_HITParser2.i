x = 5
pi = 3.14159
day = 'day'

[Models]
  [foo]
    type = SampleParserTestingModel
    bool = true
    bool_vec = 'true false false'
    bool_vec_vec = 'true false; false true true; false false false'
    int = ${x}
    int_vec = '5 6 7'
    int_vec_vec = '-1 3 -2; -5'
    uint = 30
    uint_vec = '1 2 3'
    uint_vec_vec = '555; 123; 1 5 9'
    double = ${pi}
    double_vec = '-111 12 1.1'
    double_vec_vec = '1 3 5; 2 4 6; -3 -5 -7'
    string = 'today'
    string_vec = 'is a good ${day}'
    string_vec_vec = 'neml2 is very; useful'
    shape = '(1,2,3,5)'
    shape_vec = '(1,2,3) (2,3) (5)'
    shape_vec_vec = '(2,5) () (3,3); (2,2) (1) (22)'
    suppressed_option = 'this should not work'
  []
[]
