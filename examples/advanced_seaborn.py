import seaborn as sns
import matplotlib.pyplot as plt
from plot2llm import FigureConverter

# Advanced seaborn example: FacetGrid
iris = sns.load_dataset('iris')
g = sns.FacetGrid(iris, col='species')
g.map_dataframe(sns.scatterplot, x='sepal_length', y='sepal_width')
g.fig.suptitle('Advanced Seaborn FacetGrid Example', y=1.02)

converter = FigureConverter()
print('--- TEXT OUTPUT (FacetGrid) ---')
print(converter.convert(g.fig, 'text'))
print('\n--- JSON OUTPUT (FacetGrid) ---')
print(converter.convert(g.fig, 'json'))
print('\n--- SEMANTIC OUTPUT (FacetGrid) ---')
print(converter.convert(g.fig, 'semantic'))

# Advanced seaborn example: PairPlot
pairplot = sns.pairplot(iris, hue='species')
pairplot.fig.suptitle('Advanced Seaborn PairPlot Example', y=1.02)

print('\n--- TEXT OUTPUT (PairPlot) ---')
print(converter.convert(pairplot.fig, 'text'))
print('\n--- JSON OUTPUT (PairPlot) ---')
print(converter.convert(pairplot.fig, 'json'))
print('\n--- SEMANTIC OUTPUT (PairPlot) ---')
print(converter.convert(pairplot.fig, 'semantic')) 