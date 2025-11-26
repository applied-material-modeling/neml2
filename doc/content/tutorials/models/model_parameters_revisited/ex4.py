import neml2

model = neml2.load_model("input2.i", "eq")
print(model, "\n")
print("eq1_alpha:")
print(model.eq1_alpha.tensor())
