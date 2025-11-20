import neml2

model = neml2.load_model("input.i", "my_model")

G = model.G
K = model.K

print("G:")
print(G.tensor())
print("K:")
print(K.tensor())
