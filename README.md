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

---

## Features

| Feature                        | Status           |
|--------------------------------|------------------|
| Matplotlib plots               | ✅ Full support  |
| Seaborn plots                  | ✅ Major types   |
| JSON/Text/Semantic output      | ✅               |
| Custom formatters/analyzers    | ✅               |
| Multi-axes/subplots            | ✅               |
| Level of detail control        | ✅               |
| Error handling                 | ✅               |
| Extensible API                 | ✅               |
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
  "figure_type": "matplotlib",
  "title": "Demo Plot",
  "axes": [...],
  "figure_info": {...},
  "plot_description": "This is a matplotlib visualization titled 'Demo Plot'. It contains 1 subplot(s). Subplot 1 contains: line."
}
```

---

## API Reference

See the full [API Reference](docs/API.md) for details on all classes and methods.

---

## Project Status

This project is in **beta**. Major functionalities (matplotlib, seaborn, extensibility, output formats) are stable and tested. Plotly, Bokeh, Altair, Jupyter plugin, and image-based analysis are planned but not yet implemented. We welcome contributions or feedback.

- [x] Matplotlib support
- [x] Seaborn support
- [x] Extensible formatters/analyzers
- [x] Multi-format output (text, json, semantic)
- [ ] Plotly/Bokeh/Altair integration
- [ ] Jupyter plugin
- [ ] Export to Markdown/HTML
- [ ] Image-based plot analysis

---

## Changelog / Bugfixes

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
