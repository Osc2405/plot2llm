import matplotlib.pyplot as plt
from plot2llm import FigureConverter

# Minimal matplotlib example
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [2, 4, 6], marker="o")
ax.set_title("Minimal Example")
ax.set_xlabel("X")
ax.set_ylabel("Y")

converter = FigureConverter()

print("--- TEXT OUTPUT ---")
print(converter.convert(fig, "text"))

print("\n--- JSON OUTPUT ---")
print(converter.convert(fig, "json"))

print("\n--- SEMANTIC OUTPUT ---")
print(converter.convert(fig, "semantic"))
