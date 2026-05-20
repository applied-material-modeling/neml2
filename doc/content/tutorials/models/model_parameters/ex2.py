import neml2

model = neml2.load_model("input.i", "my_model")

for pname, pval in model.named_parameters().items():
    print(f"{pname}:")
    print(pval.tensor())
