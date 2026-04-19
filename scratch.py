import json

with open("model.ipynb", "r") as f:
    nb = json.load(f)

# Find where Section 5 is.
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "## 5. K-Means Clustering" in "".join(cell["source"]):
        print("Found section 5 at index:", i)
        break

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "## 8. Profil Cluster" in "".join(cell["source"]):
        print("Found section 8 at index:", i)
        break

