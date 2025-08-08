import matplotlib.pyplot as plt
import numpy as np

from plot2llm import FigureConverter

# Advanced matplotlib example: multiple subplots and plot types
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Line plot
axs[0, 0].plot([1, 2, 3], [2, 4, 6], marker="o", label="Line")
axs[0, 0].set_title("Line Plot")
axs[0, 0].legend()

# Bar plot
axs[0, 1].bar(["A", "B", "C"], [10, 20, 15], color="skyblue")
axs[0, 1].set_title("Bar Plot")

# Scatter plot
axs[1, 0].scatter([1, 2, 3], [3, 2, 5], color="red", label="Scatter")
axs[1, 0].set_title("Scatter Plot")
axs[1, 0].legend()

# Histogram

np.random.seed(0)
data = np.random.randn(100)
axs[1, 1].hist(data, bins=15, color="green", alpha=0.7)
axs[1, 1].set_title("Histogram")

fig.suptitle("Advanced Matplotlib Example")

converter = FigureConverter()

print("--- TEXT OUTPUT ---")
print(converter.convert(fig, "text"))

print("\n--- JSON OUTPUT ---")
print(converter.convert(fig, "json"))

print("\n--- SEMANTIC OUTPUT ---")
print(converter.convert(fig, "semantic"))
