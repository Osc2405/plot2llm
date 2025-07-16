import seaborn as sns
import matplotlib.pyplot as plt
from plot2llm import FigureConverter

# Minimal seaborn example
iris = sns.load_dataset('iris')
fig, ax = plt.subplots()
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', ax=ax)
ax.set_title('Minimal Seaborn Example')

converter = FigureConverter()

print('--- TEXT OUTPUT ---')
print(converter.convert(fig, 'text'))

print('\n--- JSON OUTPUT ---')
print(converter.convert(fig, 'json'))

print('\n--- SEMANTIC OUTPUT ---')
print(converter.convert(fig, 'semantic')) 