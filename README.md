# Plot2LLM

A Python library for converting figures from popular plotting libraries (matplotlib, seaborn, plotly) into formats understandable by Large Language Models (LLMs).

## 🚀 Features

- **Multi-library Support**: Works with matplotlib, seaborn, plotly, and more
- **Comprehensive Analysis**: Extracts data, statistics, colors, and metadata
- **Multiple Output Formats**: JSON, text, and semantic formats
- **Advanced Statistics**: Per-axis and per-curve statistical analysis
- **Heatmap Support**: Full matrix data extraction for heatmaps
- **Robust Error Handling**: Graceful handling of edge cases and warnings

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

# Create a simple plot
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

- **Matplotlib**: Line plots, scatter plots, histograms, bar charts
- **Seaborn**: Heatmaps, FacetGrid, PairGrid, distribution plots
- **Plotly**: Interactive plots and charts
- **Pandas**: DataFrame plotting methods

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

- **JSON**: Structured data format for programmatic access
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

### ✅ Heatmap Processing
- **Full matrix data extraction** from seaborn heatmaps
- **QuadMesh support** for complete data access
- **Colorbar analysis** with statistical information
- **Robust handling** of masked arrays

### ✅ Numpy Warning Optimization
- **Global warning suppression** for cleaner output
- **Robust array handling** with NaN and infinite values
- **Improved statistical functions** with edge case handling

### ✅ Enhanced Axis Information
- **Detailed title extraction** for subplots
- **Axis label detection** (X and Y labels)
- **Data presence detection** per axis
- **Figure title extraction** from suptitle

### ✅ Consistent JSON Format
- **Unified structure** across all analyzers
- **Numpy array serialization** support
- **Error handling** with fallback values
- **Metadata inclusion** for analysis tracking

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

The library provides comprehensive analysis in a structured format:

```json
{
  "figure_type": "seaborn",
  "figure_info": {
    "figure_type": "matplotlib.Figure",
    "dimensions": [8.0, 6.0],
    "title": "Sample Plot",
    "axes_count": 2
  },
  "axis_info": {
    "axes": [
      {
        "index": 0,
        "title": "Main Plot",
        "x_label": "X Axis",
        "y_label": "Y Axis",
        "has_data": true
      }
    ],
    "figure_title": "Main Figure Title",
    "total_axes": 1
  },
  "colors": [
    {
      "hex": "#1f77b4",
      "name": "steel blue"
    }
  ],
  "statistics": {
    "per_axis": [
      {
        "axis_index": 0,
        "title": "Main Plot",
        "data_types": ["scatter_plot"],
        "data_points": 100,
        "mean": 0.5,
        "std": 0.3,
        "min": 0.1,
        "max": 0.9,
        "median": 0.5,
        "outliers": [],
        "skewness": 0.0,
        "kurtosis": 0.0
      }
    ],
    "per_curve": [],
    "global": {
      "mean": 0.5,
      "std": 0.3,
      "min": 0.1,
      "max": 0.9,
      "median": 0.5
    }
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of matplotlib, seaborn, and plotly
- Inspired by the need for better LLM-figure interaction
- Community contributions and feedback

## 📞 Support

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.

---

**Note**: Example outputs and test results are organized in the `plot2llm_examples/` directory (excluded from the main repository via .gitignore).