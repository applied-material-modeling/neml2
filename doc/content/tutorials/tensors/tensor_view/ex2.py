from neml2.tensors import Tensor

# Create a tensor with shape (; 4, 3) filled with zeros
a = Tensor.zeros((4, 3))
print("a =")
print(a)

# b is a view into the first row and the third row of a
b = a.base[::2]
print("b =")
print(b)

# Modification in a is reflected in b
a += 1.0
print("\nAfter first modification")
print("a =")
print(a)
print("b =")
print(b)

# Modification in data viewed by b is reflected in a
b += 1.0
print("\nAfter second modification")
print("a =")
print(a)
print("b =")
print(b)
