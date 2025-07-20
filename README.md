# plot2llm

> **Beta Notice:** This library is currently in beta. The API may change in future releases. Please report any issues or suggestions!

Convert Python figures (matplotlib, seaborn) to LLM-readable formats

---

## Introduction & Motivation

**Plot2LLM** is a Python library designed to bridge the gap between data visualization and AI. It analyzes and converts figures from matplotlib and seaborn into structured, LLM-friendly text or JSON. This enables:
- Automated documentation of plots
- Feeding plot information to Large Language Models (LLMs)
- Building explainable AI pipelines
- Generating technical summaries for reports

---

## Supported Features

- **Matplotlib support:**
  - Line, scatter, bar, histogram, box, violin, and more
  - Multi-axes and subplots
  - Extraction of axes, labels, data, colors, statistics
- **Seaborn support:**
  - Common plot types: scatter, line, box, violin, histogram, FacetGrid, PairPlot, etc.
  - Automatic detection of seaborn-specific features
- **Output formats:**
  - `'text'`: Human-readable technical summary
  - `'json'`: Structured JSON/dict
  - `'semantic'`: LLM-optimized dict with standard keys
- **Error handling:**
  - Graceful handling of invalid figures or unsupported formats
- **Extensible:**
  - Add your own formatters or analyzers

---

## Roadmap / Pending Features

- [ ] **Plotly, Bokeh, Altair support** (planned)
- [ ] **Interactive plot extraction**
- [ ] **Image-based plot analysis**
- [ ] **More advanced statistics and trend detection**
- [ ] **Better support for custom matplotlib artists**
- [ ] **Jupyter notebook integration**
- [ ] **Export to Markdown/HTML**

---

## Installation

```bash
pip install plot2llm
```

Or, for local development:

```bash
git clone <repo-url>
cd plot2llm
pip install -e .
```

---

## Quick Start

```python
import matplotlib.pyplot as plt
from plot2llm import FigureConverter

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [2, 4, 6])
ax.set_title('Demo Plot')

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

## API Reference (Summary)

### `FigureConverter`
- `convert(figure, output_format='text')`: Convert a figure to the specified format. `output_format` can be `'text'`, `'json'`, `'semantic'`, or a custom formatter object.
- `register_analyzer(name, analyzer)`: Add a custom analyzer.
- `register_formatter(name, formatter)`: Add a custom formatter.

### Formatters
- `TextFormatter`: Returns a technical, human-readable summary.
- `JSONFormatter`: Returns a structured dict (JSON-serializable).
- `SemanticFormatter`: Returns a dict optimized for LLMs, with standard keys.

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

- GitHub: [https://github.com/plot2llm/plot2llm](https://github.com/plot2llm/plot2llm)
- Issues: [https://github.com/plot2llm/plot2llm/issues](https://github.com/plot2llm/plot2llm/issues)