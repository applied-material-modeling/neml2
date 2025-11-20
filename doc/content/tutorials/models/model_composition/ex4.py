import neml2
from neml2.tensors import SR2

factory = neml2.load_input("input_composed.i")
eq1 = factory.get_model("eq1")
eq2 = factory.get_model("eq2")
eq3 = factory.get_model("eq3")
eq = factory.get_model("eq")

print("eq1:")
print(eq1, "\n")
print("eq2:")
print(eq2, "\n")
print("eq3:")
print(eq3, "\n")
print("eq:")
print(eq, "\n")
