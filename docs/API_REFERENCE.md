# API Reference - Plot2LLM

## Table of Contents

1. [Main Function](#main-function)
2. [Main Classes](#main-classes)
3. [Analyzers](#analyzers)
4. [Formatters](#formatters)
5. [Utilities](#utilities)
6. [Data Structures](#data-structures)
7. [Statistical Analysis](#statistical-analysis)

---

## Main Function

### `plot2llm.convert(figure, format='text', **kwargs)`

Main function to convert figures into LLM-optimized formats.

**Parameters:**
- `figure`: Figure from matplotlib, seaborn, plotly, etc.
- `format` (str, optional): Output format. Valid values: `'text'`, `'json'`, `'semantic'`. Default: `'text'`
- `detail_level` (str, optional): Analysis detail level. Valid values: `'low'`, `'medium'`, `'high'`. Default: `'medium'`
- `include_statistics` (bool, optional): Include statistical analysis. Default: `True`
- `include_visual_info` (bool, optional): Include visual information. Default: `True`
- `include_data_analysis` (bool, optional): Include data analysis. Default: `True`
- `include_curve_points` (bool, optional): Include raw data points for detailed analysis. Default: `False`

**Returns:**
- `str` or `dict`: Converted data in the specified format

**Example:**
```python
import plot2llm
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])

# Basic conversion
result = plot2llm.convert(fig)

# Conversion with enhanced statistical analysis
result = plot2llm.convert(
    fig,
    format='semantic',
    detail_level='high',
    include_statistics=True,
    include_visual_info=True,
    include_curve_points=True
)
```

---

## Main Classes

### `FigureConverter`

Main class for converting figures with custom configuration.

#### Constructor

```python
FigureConverter(
    detail_level="medium",
    include_data=True,
    include_colors=True,
    include_statistics=True,
    include_curve_points=False
)
```

**Parameters:**
- `detail_level` (str): Analysis detail level
- `include_data` (bool): Include figure data
- `include_colors` (bool): Include color information
- `include_statistics` (bool): Include statistical analysis
- `include_curve_points` (bool): Include raw data points for detailed analysis

#### Methods

##### `register_analyzer(name, analyzer)`

Register a custom analyzer.

**Parameters:**
- `name` (str): Analyzer name
- `analyzer`: Analyzer instance

**Example:**
```python
from plot2llm.analyzers import MatplotlibAnalyzer

converter = FigureConverter()
converter.register_analyzer('matplotlib', MatplotlibAnalyzer())
```

##### `register_formatter(name, formatter)`

Register a custom formatter.

**Parameters:**
- `name` (str): Formatter name
- `formatter`: Formatter instance

##### `convert(figure, output_format="text")`

Convert a figure to the specified format.

**Parameters:**
- `figure`: Figure to convert
- `output_format` (str): Output format

**Returns:**
- `str` or `dict`: Converted data

---

## Analyzers

### `BaseAnalyzer`

Base class for all analyzers.

#### Methods

##### `analyze(figure, **kwargs)`

Abstract method that must be implemented by subclasses.

**Parameters:**
- `figure`: Figure to analyze
- `**kwargs`: Additional arguments

**Returns:**
- `dict`: Figure analysis

### `MatplotlibAnalyzer`

Specific analyzer for matplotlib figures.

#### Constructor

```python
MatplotlibAnalyzer(
    detail_level="medium",
    include_data=True,
    include_colors=True,
    include_statistics=True
)
```

#### Methods

##### `analyze(figure, **kwargs)`

Analyze a matplotlib figure.

**Parameters:**
- `figure`: Matplotlib figure
- `detail_level` (str, optional): Detail level
- `include_statistics` (bool, optional): Include statistics
- `include_visual_info` (bool, optional): Include visual information

**Returns:**
- `dict`: Complete figure analysis

**Return structure:**
```python
{
    "figure_type": "matplotlib.figure.Figure",
    "title": "Figure title",
    "axes_count": 1,
    "dimensions": [6.4, 4.8],
    "basic_info": {
        "figure_type": "matplotlib.figure.Figure",
        "title": "Figure title",
        "axes_count": 1,
        "dimensions": [6.4, 4.8]
    },
    "axes_info": [
        {
            "title": "Axis title",
            "plot_types": [
                {
                    "type": "scatter",
                    "label": "Label",
                    "data_points": 10
                }
            ],
            "xlabel": "X Label",
            "ylabel": "Y Label",
            "x_range": [0.0, 10.0],
            "y_range": [0.0, 10.0],
            "has_grid": False,
            "has_legend": False
        }
    ],
    "data_info": {
        "data_points": 10,
        "data_types": {"x": "numeric", "y": "numeric"},
        "plot_types": [
            {
                "type": "scatter",
                "label": "Label",
                "data_points": 10
            }
        ],
        "statistics": {
            "global": {
                "mean": 5.0,
                "std": 2.5,
                "min": 0.0,
                "max": 10.0,
                "median": 5.0
            },
            "per_curve": [
                {
                    "label": "Label",
                    "mean": 5.0,
                    "std": 2.5,
                    "min": 0.0,
                    "max": 10.0,
                    "median": 5.0,
                    "trend": "stable",
                    "local_var": 0.1,
                    "outliers": []
                }
            ],
            "per_axis": [
                {
                    "axis_index": 0,
                    "title": "Axis title",
                    "mean": 5.0,
                    "std": 2.5,
                    "min": 0.0,
                    "max": 10.0,
                    "median": 5.0,
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                    "outliers": []
                }
            ]
        }
    },
    "visual_info": {
        "colors": [
            {
                "hex": "#1f77b4",
                "name": "blue",
                "rgb": [31, 119, 180]
            }
        ],
        "markers": [
            {
                "code": "o",
                "name": "circle"
            }
        ],
        "line_styles": ["solid"],
        "background_color": "white"
    }
}
```

### `SeabornAnalyzer`

Specific analyzer for seaborn figures.

#### Constructor

```python
SeabornAnalyzer(
    detail_level="medium",
    include_data=True,
    include_colors=True,
    include_statistics=True
)
```

#### Methods

##### `analyze(figure, **kwargs)`

Analyze a seaborn figure.

**Parameters:**
- `figure`: Seaborn figure
- `detail_level` (str, optional): Detail level
- `include_statistics` (bool, optional): Include statistics
- `include_visual_info` (bool, optional): Include visual information

**Returns:**
- `dict`: Complete figure analysis

**Special features:**
- Automatic heatmap detection
- FacetGrid and PairGrid analysis
- Matrix data extraction for heatmaps

---

## Formatters

### `BaseFormatter`

Base class for all formatters.

#### Methods

##### `format(analysis, **kwargs)`

Abstract method that must be implemented by subclasses.

**Parameters:**
- `analysis` (dict): Analysis data
- `**kwargs`: Additional arguments

**Returns:**
- `str` or `dict`: Formatted data

### `TextFormatter`

Formatter that converts analysis into structured text.

#### Constructor

```python
TextFormatter()
```

#### Methods

##### `format(analysis, **kwargs)`

Convert analysis to structured text.

**Parameters:**
- `analysis` (dict): Analysis data
- `**kwargs`: Additional arguments

**Returns:**
- `str`: Structured text

**Example output:**
```
Keywords in figure: scatter
Plot types in figure: scatter
Figure type: matplotlib.figure.Figure
Dimensions (inches): [6.4, 4.8]
Title: My Chart
Number of axes: 1

Axis 0: title=no_title, plot types: [scatter], xlabel: X Axis (lower: x axis), ylabel: Y Axis (lower: y axis), x_range: [1.0, 4.0], y_range: [1.0, 4.0], grid: False, legend: False

Data points: 4
Data types: {'x': 'numeric', 'y': 'numeric'}
Global statistics: mean=2.5, std=1.29, min=1.0, max=4.0, median=2.5

Colors: ['#1f77b4 (blue)']
Markers: ['o (circle)']
Line styles: ['solid']
Background color: white
```

### `JSONFormatter`

Formatter that converts analysis into structured JSON.

#### Constructor

```python
JSONFormatter()
```

#### Methods

##### `format(analysis, **kwargs)`

Convert analysis to structured JSON.

**Parameters:**
- `analysis` (dict): Analysis data
- `**kwargs`: Additional arguments

**Returns:**
- `dict`: Structured JSON

**Example output:**
```json
{
  "figure_type": "matplotlib.figure.Figure",
  "title": "My Chart",
  "axes_count": 1,
  "dimensions": [6.4, 4.8],
  "axes_info": [
    {
      "title": null,
      "plot_types": [
        {
          "type": "scatter",
          "label": null,
          "data_points": 4
        }
      ],
      "xlabel": "X Axis",
      "ylabel": "Y Axis",
      "x_range": [1.0, 4.0],
      "y_range": [1.0, 4.0],
      "has_grid": false,
      "has_legend": false
    }
  ],
  "data_info": {
    "data_points": 4,
    "data_types": {"x": "numeric", "y": "numeric"},
    "statistics": {
      "global": {
        "mean": 2.5,
        "std": 1.29,
        "min": 1.0,
        "max": 4.0,
        "median": 2.5
      }
    }
  },
  "visual_info": {
    "colors": [
      {
        "hex": "#1f77b4",
        "name": "blue",
        "rgb": [31, 119, 180]
      }
    ],
    "markers": [
      {
        "code": "o",
        "name": "circle"
      }
    ],
    "line_styles": ["solid"],
    "background_color": "white"
  }
}
```

### `SemanticFormatter`

Formatter that converts analysis into semantic format for LLMs.

#### Constructor

```python
SemanticFormatter()
```

#### Methods

##### `format(analysis, **kwargs)`

Convert analysis to semantic format.

**Parameters:**
- `analysis` (dict): Analysis data
- `**kwargs`: Additional arguments

**Returns:**
- `str`: Semantic format

**Example output:**
```
FIGURE ANALYSIS:
Type: Scatter plot with 4 data points
Content: X-axis labeled "X Axis", Y-axis labeled "Y Axis"
Data: Numeric values ranging from 1.0 to 4.0
Statistics: Mean=2.5, Standard deviation=1.29
Visual: No grid, no legend, standard scatter markers
Colors: Blue (#1f77b4)
Markers: Circle (o)
Background: White
```

---

## Utilities

### `plot2llm.utils`

Utility module for figure processing.

#### Functions

##### `detect_figure_type(figure)`

Automatically detect figure type.

**Parameters:**
- `figure`: Figure to analyze

**Returns:**
- `str`: Detected figure type

**Supported types:**
- `'matplotlib'`
- `'seaborn'`
- `'plotly'`
- `'generic'`

##### `extract_colors(artists)`

Extract color information from matplotlib artists.

**Parameters:**
- `artists`: List of matplotlib artists

**Returns:**
- `list`: List of extracted colors

##### `extract_markers(artists)`

Extract marker information from matplotlib artists.

**Parameters:**
- `artists`: List of matplotlib artists

**Returns:**
- `list`: List of extracted markers

##### `calculate_statistics(data)`

Calculate statistics from data.

**Parameters:**
- `data`: Numeric data

**Returns:**
- `dict`: Calculated statistics

---

## Data Structures

### Analysis Structure

```python
{
    "figure_type": str,           # Figure type
    "title": str,                 # Figure title
    "axes_count": int,            # Number of axes
    "dimensions": [float, float], # Dimensions in inches
    "basic_info": dict,           # Basic information
    "axes_info": list,            # Information for each axis
    "data_info": dict,            # Data information
    "visual_info": dict           # Visual information
}
```

### Axis Information Structure

```python
{
    "title": str,                 # Axis title
    "plot_types": list,           # Chart types
    "xlabel": str,                # X-axis label
    "ylabel": str,                # Y-axis label
    "x_range": [float, float],    # X-axis range
    "y_range": [float, float],    # Y-axis range
    "has_grid": bool,             # Has grid
    "has_legend": bool            # Has legend
}
```

### Chart Type Structure

```python
{
    "type": str,                  # Chart type
    "label": str,                 # Chart label
    "data_points": int            # Number of data points
}
```

### Statistics Structure

```python
{
    "global": {
        "mean": float,            # Mean
        "std": float,             # Standard deviation
        "min": float,             # Minimum value
        "max": float,             # Maximum value
        "median": float           # Median
    },
    "per_curve": list,            # Statistics per curve
    "per_axis": list              # Statistics per axis
}
```

### Visual Information Structure

```python
{
    "colors": list,               # List of colors
    "markers": list,              # List of markers
    "line_styles": list,          # Line styles
    "background_color": str       # Background color
}
```

---

## Error Handling

### Common Exceptions

#### `ValueError`
- Raised when an invalid figure is provided
- Raised when an unsupported format is specified

#### `NotImplementedError`
- Raised when an analyzer doesn't implement a required method

#### `TypeError`
- Raised when an incorrect data type is provided

### Error Handling Example

```python
import plot2llm

try:
    result = plot2llm.convert(figure, format='json')
except ValueError as e:
    print(f"Conversion error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Advanced Configuration

### Custom Analyzers

```python
from plot2llm.analyzers import MatplotlibAnalyzer

class CustomAnalyzer(MatplotlibAnalyzer):
    def analyze(self, figure, **kwargs):
        # Custom analysis
        analysis = super().analyze(figure, **kwargs)
        
        # Add custom information
        analysis['custom_field'] = 'custom_value'
        
        return analysis
```

### Custom Formatters

```python
from plot2llm.formatters import TextFormatter

class CustomFormatter(TextFormatter):
    def format(self, analysis, **kwargs):
        # Custom format
        return f"CUSTOM ANALYSIS: {analysis['figure_type']}"
```

### Registering Custom Components

```python
from plot2llm import FigureConverter

converter = FigureConverter()

# Register custom analyzer
converter.register_analyzer('custom', CustomAnalyzer())

# Register custom formatter
converter.register_formatter('custom', CustomFormatter())

# Use custom components
result = converter.convert(figure, output_format='custom')
```

---

## Statistical Analysis

### Enhanced Statistical Insights

Plot2LLM provides comprehensive statistical analysis for all plot types, including:

#### Central Tendency
```python
{
    "central_tendency": {
        "mean": float,            # Arithmetic mean
        "median": float,          # Median value
        "mode": float or null     # Mode (if applicable)
    }
}
```

#### Variability Analysis
```python
{
    "variability": {
        "standard_deviation": float,  # Standard deviation
        "variance": float,           # Variance
        "range": {
            "min": float,            # Minimum value
            "max": float             # Maximum value
        }
    }
}
```

#### Distribution Analysis (Histograms)
```python
{
    "distribution": {
        "skewness": float,          # Distribution skewness
        "kurtosis": float,          # Distribution kurtosis
        "skewness_interpretation": str,  # "approximately_symmetric", "right_skewed", "left_skewed"
        "kurtosis_interpretation": str   # "mesokurtic", "leptokurtic", "platykurtic"
    }
}
```

#### Correlation Analysis (Scatter Plots)
```python
{
    "correlations": [
        {
            "type": "pearson",      # Correlation type
            "value": float,         # Correlation coefficient
            "strength": str,        # "weak", "moderate", "strong"
            "direction": str        # "positive", "negative", "none"
        }
    ]
}
```

#### Outlier Detection
```python
{
    "outliers": {
        "detected": bool,           # Whether outliers were found
        "count": int,              # Number of outliers
        "x_outliers": int,         # Outliers in X-axis (if applicable)
        "y_outliers": int          # Outliers in Y-axis (if applicable)
    }
}
```

#### Data Quality Assessment
```python
{
    "data_quality": {
        "total_points": int,        # Total number of data points
        "missing_values": int       # Number of missing values
    }
}
```

### Statistical Analysis by Plot Type

#### Line Plots
- **Trend Analysis**: Monotonicity detection (increasing, decreasing, stable)
- **Pattern Recognition**: Smoothness and continuity analysis
- **Statistical Measures**: Central tendency and variability
- **Outlier Detection**: IQR-based outlier identification

#### Scatter Plots
- **Correlation Analysis**: Pearson correlation with strength and direction
- **Cluster Analysis**: Point distribution and clustering patterns
- **Outlier Detection**: Multivariate outlier detection
- **Shape Characteristics**: Monotonicity, smoothness, symmetry, continuity

#### Bar Plots
- **Categorical Analysis**: Category frequencies and distributions
- **Ranking Analysis**: Largest and smallest categories
- **Statistical Measures**: Central tendency for categorical data
- **Distribution Characteristics**: Uniformity and dominance analysis

#### Histograms
- **Distribution Analysis**: Skewness and kurtosis calculations
- **Shape Analysis**: Peak detection and valley identification
- **Bin Analysis**: Frequency distribution and bin characteristics
- **Statistical Measures**: Complete distribution statistics

### Using Statistical Analysis

```python
import plot2llm
import matplotlib.pyplot as plt
import numpy as np

# Create a scatter plot
fig, ax = plt.subplots()
x = np.random.normal(0, 1, 100)
y = 0.5 * x + np.random.normal(0, 0.3, 100)
ax.scatter(x, y)

# Convert with enhanced statistical analysis
converter = plot2llm.FigureConverter()
result = converter.convert(fig, format='semantic')

# Access statistical insights
statistical_insights = result['statistical_insights']

# Check correlation
if statistical_insights['correlations']:
    correlation = statistical_insights['correlations'][0]
    print(f"Correlation: {correlation['value']:.3f} ({correlation['strength']})")

# Check outliers
outliers = statistical_insights['outliers']
if outliers['detected']:
    print(f"Found {outliers['count']} outliers")

# Check distribution (for histograms)
if 'distribution' in statistical_insights:
    dist = statistical_insights['distribution']
    if dist['skewness'] is not None:
        print(f"Skewness: {dist['skewness']:.3f} ({dist['skewness_interpretation']})")
```

### Statistical Analysis Configuration

```python
# Enable detailed statistical analysis
converter = plot2llm.FigureConverter(
    detail_level='high',
    include_statistics=True
)

# Include raw data points for detailed analysis
result = converter.convert(
    fig, 
    format='semantic',
    include_curve_points=True
)
```

### Performance Considerations

- **Statistical analysis** adds minimal overhead (<100ms for typical plots)
- **Large datasets** (10k+ points) may take 1-2 seconds for full analysis
- **Memory usage** scales linearly with data size
- **Automatic cleanup** ensures no memory leaks

### Error Handling for Statistical Analysis

```python
try:
    result = converter.convert(fig, format='semantic')
    statistical_insights = result.get('statistical_insights', {})
    
    if statistical_insights:
        print("Statistical analysis completed successfully")
    else:
        print("No statistical insights available")
        
except Exception as e:
    print(f"Statistical analysis failed: {e}")
``` 