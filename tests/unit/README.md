# Guidelines on writing unit tests

In general, Catch2 poses almost zero restriction on how the test cases should be
structured and organized. However, we should try to restrain ourselves from
writing disorganized unit tests. I am coming up with the following rules as I
write the unit tests.

## Organization

The folder structure under `tests/unit` should mimic the folder structure under
`include/neml2`.

## Naming conventions

For convenience, I refer to those Catch2 *.cxx files as test files.

1. Each header file (*.h) should correspond to one and only one test file.
2. The test file's name should start with "test_".
3. The test file should inherit the header file's name, e.g.,
   "Foo.h" should have a test file called "test_Foo.cxx".
4. Each test file can contain one and only one `TEST_CASE`. The test case should
   be named as the header file name.
5. All test cases should be tagged by the residing folder relative to
   `tests/unit`. For example, test cases under "tests/unit/foo/bar/test_Baz.cxx"
   should be tagged "[foo/bar]".
6. Each test case should contain test cases for
   1. functions in the header file
   2. methods and members of a class
7. Each top-level section should correspond to one and only one function,
   method, member, or sub-class, and should be named accordingly (if possible).
8. Subsections are allowed and encouraged, especially in the case of overloaded
   functions and methods.
