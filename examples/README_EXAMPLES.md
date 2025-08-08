# Plot2LLM Usage Examples

This directory contains comprehensive examples demonstrating Plot2LLM capabilities for different use cases and output formats.

## 📁 Available Files

### Basic Examples
- **`minimal_matplotlib.py`**: Minimal matplotlib usage
- **`minimal_seaborn.py`**: Minimal seaborn usage
- **`seaborn_bar_histogram_demo.py`**: Bar and histogram demo with seaborn

### Advanced Examples
- **`advanced_matplotlib.py`**: Complex visualizations with matplotlib
- **`advanced_seaborn.py`**: Advanced visualizations with seaborn
- **`multi_plot_analysis_demo.py`**: Multiple plot analysis

### Real-World Use Cases
- **`real_world_analysis.py`**: Financial, marketing, and customer segmentation analysis
- **`llm_integration_demo.py`**: LLM integration and different output formats

### Semantic Output Examples
- **`semantic_output_matplotlib_example.py`**: Complete semantic output for matplotlib
- **`semantic_output_seaborn_example.py`**: Complete semantic output for seaborn
- **`test_semantic_formatter.py`**: Semantic formatter tests

## 🚀 How to Run

### Requirements
```bash
pip install plot2llm matplotlib seaborn numpy pandas scipy
```

### Basic Examples
```bash
# Minimal examples
python examples/minimal_matplotlib.py
python examples/minimal_seaborn.py

# Bar and histogram demo
python examples/seaborn_bar_histogram_demo.py
```

### Advanced Examples
```bash
# Complex visualizations
python examples/advanced_matplotlib.py
python examples/advanced_seaborn.py

# Multiple plot analysis
python examples/multi_plot_analysis_demo.py
```

### Real-World Use Cases
```bash
# Financial and marketing analysis
python examples/real_world_analysis.py

# LLM integration
python examples/llm_integration_demo.py
```

### Semantic Output
```bash
# Complete semantic output
python examples/semantic_output_matplotlib_example.py
python examples/semantic_output_seaborn_example.py

# Formatter tests
python examples/test_semantic_formatter.py
```

## 📊 Real-World Use Cases

### 1. Financial Analysis (`real_world_analysis.py`)
- **Stock prices**: Temporal evolution with trend line
- **Return distribution**: Histogram with statistics
- **Insights**: Volatility and trend analysis

### 2. Marketing Analysis (`real_world_analysis.py`)
- **Conversions by channel**: Bar chart with values
- **ROI by channel**: Profitability analysis
- **Cost vs conversion**: Scatter with labels
- **Cost distribution**: Pie chart

### 3. Customer Segmentation (`real_world_analysis.py`)
- **Demographic distribution**: Histograms by segment
- **Income analysis**: Boxplots and scatter
- **Correlation matrix**: Variable heatmap
- **Satisfaction by segment**: Violin plots

### 4. LLM Integration (`llm_integration_demo.py`)
- **Analysis pipeline**: Complex visualization with multiple subplots
- **Prompt generation**: Formats optimized for different LLMs
- **Format comparison**: Text, JSON, Semantic
- **Error handling**: Robustness demonstration

## 🔧 Output Formats

### Text Format
```python
from plot2llm import FigureConverter

converter = FigureConverter()
result = converter.convert(fig, 'text')
print(result)
```

### JSON Format
```python
result = converter.convert(fig, 'json')
print(result)
```

### Semantic Format (LLM-Optimized)
```python
result = converter.convert(fig, 'semantic')
print(result)
```

## 📈 Example Outputs

### Text Output
```
Plot types in figure: line
Figure type: matplotlib.Figure
Dimensions (inches): [8.0, 6.0]
Title: Demo Plot
Number of axes: 1
...
```

### JSON Output
```json
{
  "figure_type": "matplotlib",
  "title": "Demo Plot",
  "axes": [...],
  "data_summary": {...},
  "statistical_insights": {...}
}
```

### Semantic Output
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

## 🎯 Key Features Demonstrated

### Statistical Analysis
- Central tendency (mean, median, mode)
- Variability (standard deviation, variance, range)
- Correlations (Pearson, Spearman)
- Distribution analysis (skewness, kurtosis)
- Outlier detection (IQR method)

### Pattern Analysis
- Monotonicity detection
- Smoothness analysis
- Symmetry detection
- Continuity analysis

### Smart Detection
- Plot type detection (histogram vs bar vs line)
- Axis type detection (numeric vs categorical)
- Unicode support for numeric labels

## 🔍 Troubleshooting

### Common Issues
1. **Import errors**: Ensure all dependencies are installed
2. **Matplotlib backend**: Use `matplotlib.use('Agg')` for headless environments
3. **Memory issues**: Close figures with `plt.close(fig)` after analysis

### Performance Tips
- Use `detail_level='low'` for faster processing
- Set `include_curve_points=False` to reduce memory usage
- Process figures in batches for large datasets

## 📚 Additional Resources

- [API Reference](../docs/API_REFERENCE.md)
- [Installation Guide](../docs/INSTALLATION.md)
- [Testing Guide](../docs/TESTING_GUIDE.md)
- [LLM Analysis Guide](../LLM_ANALYSIS_GUIDE.md)
