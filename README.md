<p align="center">
  <img src="https://raw.githubusercontent.com/Osc2405/plot2llm/refs/heads/main/assets/logo.png" width="200" alt="plot2llm logo">
</p>

# plot2llm

[![PyPI](https://img.shields.io/pypi/v/plot2llm)](https://pypi.org/project/plot2llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/plot2llm)](https://pypi.org/project/plot2llm/)

> **Convert your Python plots into LLM-ready structured outputs — from matplotlib and seaborn.**

**Plot2LLM** bridges the gap between data visualization and AI. Instantly extract technical summaries, JSON, or LLM-optimized context from your figures for explainable AI, documentation, or RAG pipelines.

> 🧠 **Use the `'semantic'` format to generate structured context optimized for GPT, Claude or any RAG pipeline.**

**Latest Updates (v0.2.0):**
- ✅ **Enhanced Statistical Analysis**: Complete statistical insights for scatter plots including correlations, central tendency, and variability
- ✅ **Improved Axis Type Detection**: Smart detection of numeric vs categorical axes with Unicode support
- ✅ **Rich Pattern Analysis**: Detailed shape characteristics for scatter plots (monotonicity, smoothness, symmetry, continuity)
- ✅ **Comprehensive Test Suite**: All tests passing with enhanced error handling and warning suppression
- ✅ **Seaborn Integration**: Full support for Seaborn scatter plots with proper axis type detection

---

## Features

| Feature                        | Status           |
|--------------------------------|------------------|
| Matplotlib plots               | ✅ Full support  |
| Seaborn plots                  | ✅ Full support  |
| JSON/Text/Semantic output      | ✅               |
| Custom formatters/analyzers    | ✅               |
| Multi-axes/subplots            | ✅               |
| Level of detail control        | ✅               |
| Error handling                 | ✅               |
| Extensible API                 | ✅               |
| Statistical Analysis           | ✅ Enhanced     |
| Pattern Analysis              | ✅ Rich insights |
| Axis Type Detection           | ✅ Smart detection |
| Unicode Support               | ✅ Full support |
| Plotly/Bokeh/Altair detection  | 🚧 Planned      |
| Jupyter plugin                 | 🚧 Planned      |
| Export to Markdown/HTML        | 🚧 Planned      |
| Image-based plot analysis      | 🚧 Planned      |

---

## Who is this for?

- Data Scientists who want to document or explain their plots automatically
- AI engineers building RAG or explainable pipelines
- Jupyter Notebook users creating technical visualizations
- Developers generating automated reports with AI

---

## Installation

```bash
pip install plot2llm
```

Or, for local development:

```bash
git clone https://github.com/Osc2405/plot2llm.git
cd plot2llm
pip install -e .
```

---

## Quick Start

```python
import matplotlib.pyplot as plt
import numpy as np
from plot2llm import FigureConverter

x = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label="sin(x)", color="royalblue")
ax.plot(x, np.cos(x), label="cos(x)", color="orange")
ax.set_title('Sine and Cosine Waves')
ax.set_xlabel('Angle [radians]')
ax.set_ylabel('Value')
ax.legend()

converter = FigureConverter()
text_result = converter.convert(fig, 'text')
print(text_result)
```

---

## Detailed Usage

### Matplotlib Example

```python
import matplotlib.pyplot as plt
from plot2llm import FigureConverter

fig, ax = plt.subplots()
ax.bar(['A', 'B', 'C'], [10, 20, 15], color='skyblue')
ax.set_title('Bar Example')
ax.set_xlabel('Category')
ax.set_ylabel('Value')

converter = FigureConverter()
print(converter.convert(fig, 'text'))
```

### Seaborn Example

```python
import seaborn as sns
import matplotlib.pyplot as plt
from plot2llm import FigureConverter

iris = sns.load_dataset('iris')
fig, ax = plt.subplots()
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', ax=ax)
ax.set_title('Iris Scatter')

converter = FigureConverter()
print(converter.convert(fig, 'text'))
```

### Using Different Formatters

```python
from plot2llm.formatters import TextFormatter, JSONFormatter, SemanticFormatter

formatter = TextFormatter()
result = converter.convert(fig, formatter)
print(result)

formatter = JSONFormatter()
result = converter.convert(fig, formatter)
print(result)

formatter = SemanticFormatter()
result = converter.convert(fig, formatter)
print(result)
```

### Advanced Statistical Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt
from plot2llm import FigureConverter

# Create a scatter plot with correlation
fig, ax = plt.subplots()
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5
ax.scatter(x, y)
ax.set_title('Correlation Analysis')

converter = FigureConverter()
semantic_result = converter.convert(fig, 'semantic')

# Access statistical insights
stats = semantic_result['statistical_insights']
print(f"Correlation: {stats['correlations'][0]['value']:.3f}")
print(f"Strength: {stats['correlations'][0]['strength']}")

# Access pattern analysis
pattern = semantic_result['pattern_analysis']
print(f"Monotonicity: {pattern['shape_characteristics']['monotonicity']}")
print(f"Smoothness: {pattern['shape_characteristics']['smoothness']}")
```

---

## Example Outputs

**Text format:**
```
Plot types in figure: line
Figure type: matplotlib.Figure
Dimensions (inches): [8.0, 6.0]
Title: Demo Plot
Number of axes: 1
...
```

**JSON format:**
```json
{
  "figure_type": "matplotlib",
  "title": "Demo Plot",
  "axes": [...],
  ...
}
```

**Semantic format:**
```json
{
  "metadata": {
    "figure_type": "matplotlib",
    "detail_level": "medium",
    "analyzer_version": "0.1.0"
  },
  "axes": [
    {
      "title": "Demo Plot",
      "plot_types": [{"type": "line"}],
      "x_type": "numeric",
      "y_type": "numeric",
      "has_grid": true,
      "has_legend": true
    }
  ],
  "statistical_insights": {
    "central_tendency": {"mean": 0.5, "median": 0.4},
    "variability": {"standard_deviation": 0.8, "variance": 0.64},
    "correlations": [{"type": "pearson", "value": 0.95, "strength": "strong"}]
  },
  "pattern_analysis": {
    "pattern_type": "trend",
    "confidence_score": 0.9,
    "shape_characteristics": {
      "monotonicity": "increasing",
      "smoothness": "smooth",
      "symmetry": "symmetric"
    }
  }
}
```

---

## Advanced Features

### Statistical Analysis
Plot2LLM now provides comprehensive statistical analysis for scatter plots:

- **Central Tendency**: Mean, median, mode calculations
- **Variability**: Standard deviation, variance, range analysis
- **Correlations**: Pearson correlation coefficients with strength and direction
- **Data Quality**: Total points, missing values detection
- **X-axis Statistics**: Separate analysis for independent variables

### Pattern Analysis
Rich pattern recognition for scatter plots:

- **Monotonicity**: Increasing, decreasing, or mixed trends
- **Smoothness**: Smooth, piecewise, or discrete patterns
- **Symmetry**: Symmetric or asymmetric distributions
- **Continuity**: Continuous or discontinuous data patterns
- **Correlation Analysis**: Strength and direction of relationships

### Smart Axis Detection
Intelligent detection of axis types:

- **Numeric Detection**: Handles Unicode minus signs and various numeric formats
- **Categorical Detection**: Identifies discrete categories vs continuous ranges
- **Mixed Support**: Works with both Matplotlib and Seaborn plots

## API Reference

See the full [API Reference](docs/API.md) for details on all classes and methods.

---

## Project Status

This project is in **stable beta**. Core functionalities are production-ready with comprehensive test coverage. Enhanced statistical analysis, pattern recognition, and smart axis detection are now fully implemented.

- [x] Matplotlib support (Full)
- [x] Seaborn support (Full)
- [x] Extensible formatters/analyzers
- [x] Multi-format output (text, json, semantic)
- [x] Statistical analysis with correlations
- [x] Pattern analysis with shape characteristics
- [x] Smart axis type detection
- [x] Unicode support for numeric labels
- [x] Comprehensive error handling
- [ ] Plotly/Bokeh/Altair integration
- [ ] Jupyter plugin
- [ ] Export to Markdown/HTML
- [ ] Image-based plot analysis

---

## Changelog / Bugfixes

### v0.2.0 (Latest)
- ✅ **Enhanced Statistical Analysis**: Complete statistical insights for scatter plots including correlations, central tendency, and variability
- ✅ **Improved Axis Type Detection**: Smart detection of numeric vs categorical axes with Unicode minus sign support
- ✅ **Rich Pattern Analysis**: Detailed shape characteristics for scatter plots (monotonicity, smoothness, symmetry, continuity)
- ✅ **Seaborn Integration**: Full support for Seaborn scatter plots with proper axis type detection
- ✅ **Comprehensive Test Suite**: All tests passing with enhanced error handling and warning suppression
- ✅ **Unicode Support**: Proper handling of Unicode minus signs (U+2212) in numeric labels

### v0.1.x
- Fixed: Output formats like `'text'` now return the full formatted result, not just the format name
- Improved: Seaborn analyzer supports all major plot types
- Consistent: Output structure for all formatters

---

## Contributing

Pull requests and issues are welcome! Please see the [docs/](docs/) folder for API reference and contribution guidelines.

---

## License

MIT License

---

## Contact & Links

- [GitHub repo](https://github.com/Osc2405/plot2llm)
- [Issues](https://github.com/Osc2405/plot2llm/issues)

---

*Try it, give feedback, or suggest a formatter you’d like to see!*
