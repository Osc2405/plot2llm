# Usage Examples - Plot2LLM

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Matplotlib Examples](#matplotlib-examples)
3. [Seaborn Examples](#seaborn-examples)
4. [Plotly Examples](#plotly-examples)
5. [Advanced Analysis](#advanced-analysis)
6. [LLM Integration](#llm-integration)
7. [Real Use Cases](#real-use-cases)

---

## Basic Examples

### Example 1: Simple Conversion

```python
import matplotlib.pyplot as plt
import plot2llm

# Create a simple figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 3, 5])
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Simple Line Chart')

# Convert to text
text_result = plot2llm.convert(fig, format='text')
print("=== TEXT RESULT ===")
print(text_result)

# Convert to JSON
json_result = plot2llm.convert(fig, format='json')
print("\n=== JSON RESULT ===")
print(json_result)

plt.close()
```

### Example 2: Multiple Formats

```python
import matplotlib.pyplot as plt
import plot2llm

# Create figure
fig, ax = plt.subplots()
ax.scatter([1, 2, 3, 4], [1, 4, 2, 3], c='red', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Scatter Plot')

# Test all formats
formats = ['text', 'json', 'semantic']

for fmt in formats:
    result = plot2llm.convert(fig, format=fmt)
    print(f"\n=== FORMAT: {fmt.upper()} ===")
    print(result)

plt.close()
```

---

## Matplotlib Examples

### Example 3: Bar Chart

```python
import matplotlib.pyplot as plt
import plot2llm
import numpy as np

# Create data
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Bar Chart with Colors')

# Add values on bars
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            str(value), ha='center', va='bottom')

# Analyze with high detail
result = plot2llm.convert(
    fig, 
    format='json',
    detail_level='high',
    include_statistics=True,
    include_visual_info=True
)

print("=== DETAILED ANALYSIS ===")
print(f"Figure type: {result['figure_type']}")
print(f"Number of axes: {result['axes_count']}")
print(f"Chart types: {[pt['type'] for pt in result['axes_info'][0]['plot_types']]}")
print(f"Colors used: {len(result['visual_info']['colors'])}")
print(f"Global statistics: {result['data_info']['statistics']['global']}")

plt.close()
```

### Example 4: Histogram

```python
import matplotlib.pyplot as plt
import plot2llm
import numpy as np

# Generate random data
data = np.random.normal(100, 15, 1000)

# Create histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.set_xlabel('Values')
ax.set_ylabel('Frequency')
ax.set_title('Normal Distribution')

# Add density line
from scipy.stats import norm
x = np.linspace(data.min(), data.max(), 100)
y = norm.pdf(x, data.mean(), data.std()) * len(data) * (data.max() - data.min()) / 30
ax.plot(x, y, 'r-', linewidth=2, label='Theoretical distribution')
ax.legend()

# Analyze
result = plot2llm.convert(fig, format='text')
print("=== HISTOGRAM ANALYSIS ===")
print(result)

plt.close()
```

### Example 5: Multiple Subplots

```python
import matplotlib.pyplot as plt
import plot2llm
import numpy as np

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Complete Data Analysis', fontsize=16)

# Subplot 1: Line
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
axes[0, 0].plot(x, y1, 'b-', label='Sine')
axes[0, 0].set_title('Sine Function')
axes[0, 0].legend()

# Subplot 2: Scatter
x2 = np.random.rand(50)
y2 = np.random.rand(50)
axes[0, 1].scatter(x2, y2, c='red', alpha=0.6)
axes[0, 1].set_title('Random Scatter')

# Subplot 3: Bars
categories = ['A', 'B', 'C']
values = [10, 20, 15]
axes[1, 0].bar(categories, values, color=['green', 'orange', 'purple'])
axes[1, 0].set_title('Bar Chart')

# Subplot 4: Histogram
data = np.random.normal(0, 1, 1000)
axes[1, 1].hist(data, bins=20, alpha=0.7, color='skyblue')
axes[1, 1].set_title('Normal Distribution')

plt.tight_layout()

# Analyze complete figure
result = plot2llm.convert(
    fig, 
    format='json',
    detail_level='high'
)

print("=== MULTIPLE SUBPLOTS ANALYSIS ===")
print(f"Total number of axes: {result['axes_count']}")
for i, axis_info in enumerate(result['axes_info']):
    plot_types = [pt['type'] for pt in axis_info['plot_types']]
    print(f"Axis {i}: {plot_types}")

plt.close()
```

---

## Seaborn Examples

### Example 6: Heatmap

```python
import seaborn as sns
import plot2llm
import numpy as np
import matplotlib.pyplot as plt

# Create correlation data
np.random.seed(42)
data = np.random.rand(10, 10)
correlation_matrix = np.corrcoef(data)

# Create heatmap
fig = plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    cbar_kws={'shrink': 0.8}
)
plt.title('Correlation Matrix')

# Analyze heatmap
result = plot2llm.convert(
    fig, 
    format='json',
    detail_level='high'
)

print("=== HEATMAP ANALYSIS ===")
print(f"Figure type: {result['figure_type']}")
print(f"Data extracted: {result['data_info']['data_points']} points")
if 'matrix_data' in result['data_info']:
    print(f"Matrix shape: {result['data_info']['matrix_data']['shape']}")

plt.close()
```

### Example 7: Seaborn Scatter Plot

```python
import seaborn as sns
import plot2llm
import pandas as pd
import numpy as np

# Create dataset
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.rand(100),
    'y': np.random.rand(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.rand(100) * 100
})

# Create scatter plot
fig = plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='category',
    size='size',
    sizes=(50, 200),
    alpha=0.7
)
plt.title('Scatter Plot with Categories and Sizes')

# Analyze
result = plot2llm.convert(fig, format='text')
print("=== SEABORN SCATTER ANALYSIS ===")
print(result)

plt.close()
```

### Example 8: FacetGrid

```python
import seaborn as sns
import plot2llm
import pandas as pd
import numpy as np

# Create dataset
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.rand(200),
    'y': np.random.rand(200),
    'category': np.random.choice(['A', 'B'], 200),
    'subcategory': np.random.choice(['X', 'Y'], 200)
})

# Create FacetGrid
fig = plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df, col='category', row='subcategory', height=4, aspect=1.5)
g.map_dataframe(sns.scatterplot, x='x', y='y', alpha=0.7)
g.fig.suptitle('FacetGrid with Scatter Plot', y=1.02)

# Analyze
result = plot2llm.convert(
    fig, 
    format='json',
    detail_level='high'
)

print("=== FACETGRID ANALYSIS ===")
print(f"Number of axes: {result['axes_count']}")
print(f"Detected chart types: {[pt['type'] for ax in result['axes_info'] for pt in ax['plot_types']]}")

plt.close()
```

---

## Plotly Examples

### Example 9: Plotly Chart

```python
import plotly.graph_objects as go
import plot2llm
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='lines+markers',
    name='Sine',
    line=dict(color='blue', width=2),
    marker=dict(size=6)
))
fig.update_layout(
    title='Sine Function with Plotly',
    xaxis_title='X',
    yaxis_title='Y',
    width=800,
    height=600
)

# Analyze (Plotly converts to matplotlib internally)
result = plot2llm.convert(fig, format='text')
print("=== PLOTLY ANALYSIS ===")
print(result)
```

---

## Advanced Analysis

### Example 10: Detailed Statistical Analysis

```python
import matplotlib.pyplot as plt
import plot2llm
import numpy as np
from scipy import stats

# Create data with outliers
np.random.seed(42)
normal_data = np.random.normal(100, 15, 95)
outliers = np.random.uniform(150, 200, 5)
data = np.concatenate([normal_data, outliers])

# Create figure with multiple visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Complete Statistical Analysis', fontsize=16)

# Subplot 1: Histogram
axes[0, 0].hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Data Distribution')
axes[0, 0].set_xlabel('Values')
axes[0, 0].set_ylabel('Frequency')

# Subplot 2: Box plot
axes[0, 1].boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
axes[0, 1].set_title('Box Plot')
axes[0, 1].set_ylabel('Values')

# Subplot 3: Q-Q plot
stats.probplot(data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality)')

# Subplot 4: Density plot
from scipy.stats import gaussian_kde
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 200)
axes[1, 1].plot(x_range, kde(x_range), 'r-', linewidth=2)
axes[1, 1].set_title('Probability Density')
axes[1, 1].set_xlabel('Values')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()

# Analyze with high detail level
result = plot2llm.convert(
    fig, 
    format='json',
    detail_level='high',
    include_statistics=True,
    include_visual_info=True
)

print("=== ADVANCED STATISTICAL ANALYSIS ===")
print(f"Global statistics: {result['data_info']['statistics']['global']}")

# Show detected outliers
for i, axis_stats in enumerate(result['data_info']['statistics']['per_axis']):
    if axis_stats.get('outliers'):
        print(f"Axis {i}: {len(axis_stats['outliers'])} outliers detected")

plt.close()
```

### Example 11: Trend Analysis

```python
import matplotlib.pyplot as plt
import plot2llm
import numpy as np
from scipy import stats

# Create data with trend
np.random.seed(42)
x = np.linspace(0, 100, 50)
trend = 0.5 * x + 20
noise = np.random.normal(0, 5, 50)
y = trend + noise

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
ax.scatter(x, y, alpha=0.7, color='blue', s=50, label='Data')

# Trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line = slope * x + intercept
ax.plot(x, line, 'r-', linewidth=2, label=f'Trend (r²={r_value**2:.3f})')

# Configure plot
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Trend Analysis')
ax.legend()
ax.grid(True, alpha=0.3)

# Analyze
result = plot2llm.convert(
    fig, 
    format='json',
    detail_level='high',
    include_statistics=True
)

print("=== TREND ANALYSIS ===")
print(f"Chart types: {[pt['type'] for pt in result['axes_info'][0]['plot_types']]}")
print(f"Statistics per curve:")
for curve in result['data_info']['statistics']['per_curve']:
    print(f"  - Trend: {curve.get('trend', 'N/A')}")
    print(f"  - Local variability: {curve.get('local_var', 'N/A')}")

plt.close()
```

---

## LLM Integration

### Example 12: OpenAI Integration

```python
import matplotlib.pyplot as plt
import plot2llm
import openai
import json

# Configure OpenAI (requires API key)
# openai.api_key = "your-api-key"

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]
ax.plot(x, y, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Sales (thousands)')
ax.set_title('Weekly Sales')
ax.grid(True, alpha=0.3)

# Convert to text for LLM
figure_description = plot2llm.convert(fig, format='text')

# Prompt for LLM
prompt = f"""
Analyze the following figure description and provide insights:

{figure_description}

Please provide:
1. An executive summary of the data
2. Pattern or trend identification
3. Recommendations based on the data
4. Possible actions to take
"""

# Send to OpenAI (commented to avoid costs)
# response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#         {"role": "system", "content": "You are an expert data analyst."},
#         {"role": "user", "content": prompt}
#     ]
# )
# print(response.choices[0].message.content)

print("=== LLM DESCRIPTION ===")
print(figure_description)

plt.close()
```

### Example 13: Comparative Analysis

```python
import matplotlib.pyplot as plt
import plot2llm
import numpy as np

# Create multiple figures for comparison
figures = []

# Figure 1: Data with positive trend
fig1, ax1 = plt.subplots(figsize=(8, 6))
x1 = np.linspace(0, 10, 20)
y1 = 2 * x1 + np.random.normal(0, 1, 20)
ax1.plot(x1, y1, 'bo-')
ax1.set_title('Data with Positive Trend')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
figures.append(fig1)

# Figure 2: Data with negative trend
fig2, ax2 = plt.subplots(figsize=(8, 6))
x2 = np.linspace(0, 10, 20)
y2 = -1.5 * x2 + 20 + np.random.normal(0, 1, 20)
ax2.plot(x2, y2, 'ro-')
ax2.set_title('Data with Negative Trend')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
figures.append(fig2)

# Figure 3: Data without clear trend
fig3, ax3 = plt.subplots(figsize=(8, 6))
x3 = np.linspace(0, 10, 20)
y3 = 10 + np.random.normal(0, 2, 20)
ax3.plot(x3, y3, 'go-')
ax3.set_title('Data Without Clear Trend')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
figures.append(fig3)

# Analyze all figures
analyses = []
for i, fig in enumerate(figures):
    analysis = plot2llm.convert(
        fig, 
        format='json',
        detail_level='high',
        include_statistics=True
    )
    analyses.append(analysis)
    print(f"\n=== ANALYSIS FIGURE {i+1} ===")
    print(f"Detected trend: {analysis['data_info']['statistics']['per_curve'][0].get('trend', 'N/A')}")
    print(f"Data range: {analysis['axes_info'][0]['y_range']}")

# Compare trends
print("\n=== TREND COMPARISON ===")
for i, analysis in enumerate(analyses):
    trend = analysis['data_info']['statistics']['per_curve'][0].get('trend', 'N/A')
    print(f"Figure {i+1}: {trend}")

# Close all figures
for fig in figures:
    plt.close()
```

---

## Real Use Cases

### Example 14: Sales Analysis

```python
import matplotlib.pyplot as plt
import plot2llm
import pandas as pd
import numpy as np

# Simulate sales data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=52, freq='W')
sales_data = {
    'date': dates,
    'sales': np.random.normal(1000, 200, 52) + np.sin(np.linspace(0, 4*np.pi, 52)) * 100,
    'expenses': np.random.normal(800, 150, 52),
    'customers': np.random.poisson(50, 52)
}
df = pd.DataFrame(sales_data)

# Create analysis figure
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Annual Sales Analysis', fontsize=16)

# Subplot 1: Sales by week
axes[0, 0].plot(df['date'], df['sales'], 'b-', linewidth=2)
axes[0, 0].set_title('Weekly Sales')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Subplot 2: Sales vs Expenses
axes[0, 1].scatter(df['sales'], df['expenses'], alpha=0.7, c='red')
axes[0, 1].set_title('Sales vs Expenses')
axes[0, 1].set_xlabel('Sales ($)')
axes[0, 1].set_ylabel('Expenses ($)')

# Subplot 3: Sales distribution
axes[1, 0].hist(df['sales'], bins=15, alpha=0.7, color='green', edgecolor='black')
axes[1, 0].set_title('Sales Distribution')
axes[1, 0].set_xlabel('Sales ($)')
axes[1, 0].set_ylabel('Frequency')

# Subplot 4: Customers by week
axes[1, 1].bar(range(len(df)), df['customers'], alpha=0.7, color='orange')
axes[1, 1].set_title('Customers by Week')
axes[1, 1].set_xlabel('Week')
axes[1, 1].set_ylabel('Number of Customers')

plt.tight_layout()

# Analyze
result = plot2llm.convert(
    fig, 
    format='text',
    detail_level='high'
)

print("=== SALES ANALYSIS ===")
print(result)

plt.close()
```

### Example 15: Scientific Data Analysis

```python
import matplotlib.pyplot as plt
import plot2llm
import numpy as np
from scipy import stats

# Simulate scientific data (temperature experiment)
np.random.seed(42)
temperatures = np.linspace(20, 80, 30)
# Non-linear response with noise
response = 100 * np.exp(-0.05 * (temperatures - 50)**2) + np.random.normal(0, 5, 30)

# Create scientific figure
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Temperature Response Analysis', fontsize=16)

# Subplot 1: Response curve
axes[0, 0].scatter(temperatures, response, c='blue', alpha=0.7, s=60)
axes[0, 0].set_xlabel('Temperature (°C)')
axes[0, 0].set_ylabel('Response (units)')
axes[0, 0].set_title('Response Curve')
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Residuals
# Polynomial fit
coeffs = np.polyfit(temperatures, response, 2)
fitted = np.polyval(coeffs, temperatures)
residuals = response - fitted

axes[0, 1].scatter(temperatures, residuals, c='red', alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='--')
axes[0, 1].set_xlabel('Temperature (°C)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Analysis')
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Q-Q plot of residuals
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot of Residuals')

# Subplot 4: Residual histogram
axes[1, 1].hist(residuals, bins=10, alpha=0.7, color='green', edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')

plt.tight_layout()

# Analyze
result = plot2llm.convert(
    fig, 
    format='json',
    detail_level='high',
    include_statistics=True
)

print("=== SCIENTIFIC ANALYSIS ===")
print(f"Detected chart types: {[pt['type'] for ax in result['axes_info'] for pt in ax['plot_types']]}")
print(f"Global statistics: {result['data_info']['statistics']['global']}")

plt.close()
```

---

## Tips and Best Practices

### 1. Performance Optimization

```python
# For analysis of many figures, use low detail level
result = plot2llm.convert(fig, detail_level='low')

# For detailed analysis, use high level
result = plot2llm.convert(fig, detail_level='high')
```

### 2. Error Handling

```python
import plot2llm

try:
    result = plot2llm.convert(figure)
except ValueError as e:
    print(f"Conversion error: {e}")
    # Handle specific error
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle general error
```

### 3. Customization

```python
# Create custom analyzer
from plot2llm.analyzers import MatplotlibAnalyzer

class CustomAnalyzer(MatplotlibAnalyzer):
    def analyze(self, figure, **kwargs):
        analysis = super().analyze(figure, **kwargs)
        # Add custom information
        analysis['custom_metric'] = self.calculate_custom_metric(figure)
        return analysis
    
    def calculate_custom_metric(self, figure):
        # Custom logic
        return "custom_value"
```

### 4. Integration in Pipelines

```python
def analyze_figure_pipeline(figure, output_format='json'):
    """Complete figure analysis pipeline"""
    
    # Step 1: Validate figure
    if figure is None:
        raise ValueError("Figure cannot be None")
    
    # Step 2: Analyze
    result = plot2llm.convert(
        figure, 
        format=output_format,
        detail_level='high',
        include_statistics=True
    )
    
    # Step 3: Post-processing
    if output_format == 'json':
        result['analysis_timestamp'] = pd.Timestamp.now().isoformat()
        result['figure_hash'] = hash(str(figure))
    
    return result
```

---

## Conclusion

These examples demonstrate the versatility and power of Plot2LLM for different types of analysis. The library can adapt to multiple use cases, from basic analysis to complex scientific applications.

For more information, consult the [API Reference](API_REFERENCE.md) and the [main README](../README.md). 