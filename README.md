# Plot2LLM

> ⚠️ **This library is under active development. Currently, only Matplotlib and Seaborn are supported. Support for other libraries such as Plotly, Bokeh, Altair, and Pandas is planned but not yet implemented.**

A Python library for converting figures from popular visualization libraries (matplotlib, seaborn, plotly) into formats understandable by Large Language Models (LLMs).

## 🚀 Features

- **Multi-library support**: Currently Matplotlib and Seaborn
- **Comprehensive analysis**: Extracts data, statistics, colors, and metadata
- **Multiple output formats**: JSON, text, and semantic
- **Advanced statistics**: Per-axis and per-curve statistical analysis
- **Heatmap support**: Full matrix data extraction for heatmaps
- **Robust error handling**: Handles edge cases and warnings gracefully

## 📦 Installation

```bash
pip install plot2llm
```

Or install from source:

```bash
git clone https://github.com/yourusername/plot2llm.git
cd plot2llm
pip install -e .
```

## 🎯 Quick Start

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plot2llm import FigureConverter

# Create a simple heatmap
data = np.random.rand(10, 10)
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, cmap='viridis')
plt.title("Sample Heatmap")

# Convert to LLM format
converter = FigureConverter()
result = converter.convert(plt.gcf(), output_format="json")

# Access the analysis
print(f"Figure type: {result['figure_type']}")
print(f"Total axes: {result['axis_info']['total_axes']}")
print(f"Matrix data shape: {result['statistics']['per_axis'][0]['matrix_data']['shape']}")
```

## 📊 Supported Libraries

- **Matplotlib**: Line plots, scatter plots, histograms, bar charts, etc.
- **Seaborn**: Heatmaps, FacetGrid, PairGrid, distribution plots

**Planned (not yet implemented):**
- Plotly
- Bokeh
- Altair
- Pandas

## 🛣️ Roadmap / Next Steps

- [ ] Support for Plotly figures
- [ ] Support for Bokeh
- [ ] Support for Altair
- [ ] Support for Pandas plots
- [ ] Improved documentation and examples
- [ ] More tests and cross-validation

Contributions to add support for new libraries are welcome!

## 🔧 API Reference

### FigureConverter

Main class for converting figures to LLM-readable formats.

```python
converter = FigureConverter(
    detail_level="medium",      # "low", "medium", "high"
    include_data=True,          # Include data statistics
    include_colors=True,        # Include color information
    include_statistics=True     # Include statistical analysis
)
```

### Output Formats

- **JSON**: Structured format for programmatic access
- **Text**: Human-readable text description
- **Semantic**: Rich semantic structure for LLM processing

## 📁 Project Structure

```
plot2llm/
├── plot2llm/
│   ├── __init__.py
│   ├── converter.py          # Main converter class
│   ├── formatters.py         # Output format handlers
│   ├── utils.py              # Utility functions
│   └── analyzers/
│       ├── __init__.py
│       ├── base_analyzer.py  # Base analyzer class
│       ├── matplotlib_analyzer.py
│       └── seaborn_analyzer.py
├── tests/                    # Unit tests
├── examples/                 # Usage examples
├── docs/                     # Documentation
└── README.md
```

## 🧪 Examples

### Basic Usage

```python
from plot2llm import FigureConverter
import matplotlib.pyplot as plt
import seaborn as sns

# Create a plot
sns.scatterplot(data=df, x='x', y='y', hue='category')
plt.title("Sample Scatter Plot")

# Convert to JSON
converter = FigureConverter()
result = converter.convert(plt.gcf(), output_format="json")

# Access results
print(f"Figure type: {result['figure_type']}")
print(f"Colors used: {len(result['colors'])}")
print(f"Statistics: {result['statistics']['per_axis']}")
```

### Advanced Analysis

```python
# High detail level with all features
converter = FigureConverter(
    detail_level="high",
    include_data=True,
    include_colors=True,
    include_statistics=True
)

result = converter.convert(figure, output_format="semantic")

# Access detailed information
axis_info = result['axis_info']
for axis in axis_info['axes']:
    print(f"Axis {axis['index']}: {axis['title']}")
    print(f"  X Label: {axis['x_label']}")
    print(f"  Y Label: {axis['y_label']}")
    print(f"  Has data: {axis['has_data']}")
```

## 🔍 Recent Improvements

- Full matrix data extraction in seaborn heatmaps
- QuadMesh support
- Colorbar and statistics analysis
- Robust handling of arrays with NaN/infinite values
- Detailed extraction of titles and axis labels
- Unified JSON format and array serialization

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific examples:

```bash
python example_seaborn.py
python test_improvements.py
```

## 📈 Output Structure

The library provides comprehensive analysis in a structured format (see examples in `/plot2llm_examples/`).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-library-support`)
3. Make your changes and commit (`git commit -m 'Add support for X'`)
4. Push to your branch (`git push origin feature/new-library-support`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of matplotlib and seaborn
- Inspired by the need for better LLM-figure interaction
- Contributions and feedback are welcome!

## 📞 Support

For questions, issues, or contributions, open an issue on GitHub or contact the maintainers.

---

**Note**: Example outputs and test results are organized in the `plot2llm_examples/` directory (excluded from the main repository via .gitignore).