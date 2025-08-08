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
- ✅ **Complete Statistical Insights**: Full distribution analysis, correlations, outliers, and central tendency for all plot types
- ✅ **Enhanced Plot Type Detection**: Improved histogram vs bar vs line detection with proper prioritization
- ✅ **Rich Pattern Analysis**: Detailed shape characteristics and pattern recognition for all visualization types
- ✅ **Comprehensive Test Suite**: 172/174 tests passing (98.9% success rate) with 24s execution time
- ✅ **Production Ready**: All core features validated with extensive error handling and edge case coverage
- ✅ **Code Quality**: All linting issues resolved with ruff and black formatting

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
| Statistical Analysis           | ✅ Complete     |
| Pattern Analysis              | ✅ Rich insights |
| Axis Type Detection           | ✅ Smart detection |
| Unicode Support               | ✅ Full support |
| Distribution Analysis         | ✅ Skewness/Kurtosis |
| Correlation Analysis          | ✅ Pearson/Spearman |
| Outlier Detection            | ✅ IQR method |
| Plot Type Detection          | ✅ Histogram/Bar/Line |
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
- Researchers needing statistical analysis of visualizations

---

## Installation

```bash
pip install plot2llm
```

For full functionality with matplotlib and seaborn:
```bash
pip install plot2llm[all]
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

## Examples

### Basic Usage
```python
import matplotlib.pyplot as plt
from plot2llm import FigureConverter

fig, ax = plt.subplots()
ax.bar(['A', 'B', 'C'], [10, 20, 15], color='skyblue')
ax.set_title('Bar Example')

converter = FigureConverter()
print(converter.convert(fig, 'text'))
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
```

### Real-World Examples
The `examples/` directory contains comprehensive examples:

- **`minimal_matplotlib.py`**: Basic matplotlib usage
- **`minimal_seaborn.py`**: Basic seaborn usage  
- **`real_world_analysis.py`**: Financial, marketing, and customer segmentation analysis
- **`llm_integration_demo.py`**: LLM integration and format comparison
- **`semantic_output_*.py`**: Complete semantic output examples

Run any example with:
```bash
python examples/minimal_matplotlib.py
```

---

## Output Formats

### Text Format
```
Plot types in figure: line
Figure type: matplotlib.Figure
Dimensions (inches): [8.0, 6.0]
Title: Demo Plot
Number of axes: 1
...
```

### JSON Format
```json
{
  "figure_type": "matplotlib",
  "title": "Demo Plot",
  "axes": [...],
  ...
}
```

### Semantic Format (LLM-Optimized)
```json
{
  "metadata": {
    "figure_type": "matplotlib",
    "detail_level": "medium"
  },
  "axes": [
    {
      "title": "Demo Plot",
      "plot_types": [{"type": "line"}],
      "x_type": "numeric",
      "y_type": "numeric"
    }
  ],
  "statistical_insights": {
    "central_tendency": {"mean": 0.5, "median": 0.4},
    "correlations": [{"type": "pearson", "value": 0.95, "strength": "strong"}]
  },
  "pattern_analysis": {
    "pattern_type": "trend",
    "shape_characteristics": {
      "monotonicity": "increasing",
      "smoothness": "smooth"
    }
  }
}
```

---

## Advanced Features

### Statistical Analysis
- **Central Tendency**: Mean, median, mode calculations
- **Variability**: Standard deviation, variance, range analysis
- **Correlations**: Pearson correlation coefficients with strength and direction
- **Data Quality**: Total points, missing values detection
- **Distribution Analysis**: Skewness and kurtosis for histograms

### Pattern Analysis
- **Monotonicity**: Increasing, decreasing, or mixed trends
- **Smoothness**: Smooth, piecewise, or discrete patterns
- **Symmetry**: Symmetric or asymmetric distributions
- **Continuity**: Continuous or discontinuous data patterns

### Smart Axis Detection
- **Numeric Detection**: Handles Unicode minus signs and various numeric formats
- **Categorical Detection**: Identifies discrete categories vs continuous ranges
- **Mixed Support**: Works with both Matplotlib and Seaborn plots

---

## API Reference

See the full [API Reference](docs/API_REFERENCE.md) for details on all classes and methods.

---

## Project Status

This project is in **stable beta**. Core functionalities are production-ready with comprehensive test coverage.

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

## Changelog

### v0.2.0 (Latest)
- ✅ **Enhanced Statistical Analysis**: Complete statistical insights for all plot types
- ✅ **Improved Plot Type Detection**: Better histogram vs bar vs line detection
- ✅ **Rich Pattern Analysis**: Detailed shape characteristics for all visualization types
- ✅ **Comprehensive Test Suite**: 172/174 tests passing (98.9% success rate)

### v0.2.0
- ✅ **Enhanced Statistical Analysis**: Complete statistical insights for all plot types
- ✅ **Improved Plot Type Detection**: Better histogram vs bar vs line detection
- ✅ **Rich Pattern Analysis**: Detailed shape characteristics for all visualization types
- ✅ **Comprehensive Test Suite**: 172/174 tests passing (98.9% success rate)

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

*Try it, give feedback, or suggest a formatter you'd like to see!*
