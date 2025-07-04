# Plot2LLM

[![PyPI version](https://badge.fury.io/py/plot2llm.svg)](https://badge.fury.io/py/plot2llm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Plot2LLM** is a Python library that converts figures from visualization libraries (matplotlib, seaborn, plotly) into structured formats optimized for Large Language Models (LLMs). It extracts detailed technical information, statistics, and metadata from figures to facilitate analysis and processing by LLMs.

## 🚀 Key Features

- **Intelligent Analysis**: Automatically detects chart types (scatter, line, bar, histogram, heatmap, etc.)
- **Multiple Formats**: Structured text, JSON, and semantic format for LLMs
- **Advanced Statistics**: Trend analysis, correlations, outliers, and distribution analysis
- **Multi-library Support**: matplotlib, seaborn, plotly and more
- **Metadata Extraction**: Colors, markers, line styles, data ranges
- **Simple API**: Single function to convert any figure

## 📦 Installation

```bash
pip install plot2llm
```

### Optional Dependencies

For full functionality, install visualization libraries:

```bash
pip install matplotlib seaborn plotly
```

## 🎯 Quick Start

```python
import matplotlib.pyplot as plt
import plot2llm

# Create a figure
fig, ax = plt.subplots()
ax.scatter([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_title('My Chart')

# Convert to text for LLMs
text_description = plot2llm.convert(fig, format='text')
print(text_description)

# Convert to structured JSON
json_data = plot2llm.convert(fig, format='json')
print(json_data)
```

## 📚 Complete Guide

### 1. Basic Conversion

```python
import plot2llm

# Simple conversion
result = plot2llm.convert(figure, format='text')

# Available formats
text_result = plot2llm.convert(figure, format='text')      # Structured text
json_result = plot2llm.convert(figure, format='json')      # Structured JSON
semantic_result = plot2llm.convert(figure, format='semantic')  # Semantic format
```

### 2. Detailed Analysis

```python
# Analysis with custom detail level
result = plot2llm.convert(
    figure, 
    format='json',
    detail_level='high',  # 'low', 'medium', 'high'
    include_statistics=True,
    include_visual_info=True
)
```

### 3. Working with Different Libraries

#### Matplotlib
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
result = plot2llm.convert(fig)
```

#### Seaborn
```python
import seaborn as sns

# Heatmap
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
fig = sns.heatmap(data).get_figure()
result = plot2llm.convert(fig)

# Scatter plot
fig = sns.scatterplot(data=df, x='x', y='y').get_figure()
result = plot2llm.convert(fig)
```

#### Plotly
```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))
result = plot2llm.convert(fig)
```

## 🔧 API Reference

### Main Function

#### `plot2llm.convert(figure, format='text', **kwargs)`

Converts a figure to the specified format.

**Parameters:**
- `figure`: Figure from matplotlib, seaborn, plotly, etc.
- `format` (str): Output format ('text', 'json', 'semantic')
- `detail_level` (str): Detail level ('low', 'medium', 'high')
- `include_statistics` (bool): Include statistical analysis
- `include_visual_info` (bool): Include visual information

**Returns:**
- `str` or `dict`: Converted data in the specified format

### Main Classes

#### `FigureConverter`
```python
from plot2llm import FigureConverter

converter = FigureConverter()
converter.register_analyzer('matplotlib', MatplotlibAnalyzer())
converter.register_formatter('custom', CustomFormatter())
result = converter.convert(figure, format='custom')
```

#### `MatplotlibAnalyzer`
```python
from plot2llm.analyzers import MatplotlibAnalyzer

analyzer = MatplotlibAnalyzer()
analysis = analyzer.analyze(figure, detail_level='high')
```

#### `TextFormatter`
```python
from plot2llm.formatters import TextFormatter

formatter = TextFormatter()
text = formatter.format(analysis_data)
```

## 📊 Output Formats

### 1. Text Format

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
```

### 2. JSON Format

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
  }
}
```

### 3. Semantic Format

```
FIGURE ANALYSIS:
Type: Scatter plot with 4 data points
Content: X-axis labeled "X Axis", Y-axis labeled "Y Axis"
Data: Numeric values ranging from 1.0 to 4.0
Statistics: Mean=2.5, Standard deviation=1.29
Visual: No grid, no legend, standard scatter markers
```

## 🎯 Real-World Use Cases

### 1. Data Science Workflow Integration

```python
import plot2llm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def automated_data_analysis(df, target_column):
    """Complete automated analysis pipeline for data science projects"""
    
    analysis_results = {}
    
    # 1. Distribution Analysis
    fig_dist, ax_dist = plt.subplots(2, 2, figsize=(12, 10))
    fig_dist.suptitle('Data Distribution Analysis', fontsize=14)
    
    # Target variable distribution
    ax_dist[0, 0].hist(df[target_column], bins=20, alpha=0.7)
    ax_dist[0, 0].set_title(f'{target_column} Distribution')
    
    # Correlation heatmap
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_dist[0, 1])
    ax_dist[0, 1].set_title('Correlation Matrix')
    
    # Box plot for outliers
    df.boxplot(column=target_column, ax=ax_dist[1, 0])
    ax_dist[1, 0].set_title('Outlier Detection')
    
    # Q-Q plot for normality
    from scipy import stats
    stats.probplot(df[target_column], dist="norm", plot=ax_dist[1, 1])
    ax_dist[1, 1].set_title('Normality Test')
    
    # Analyze the complete figure
    analysis_results['distribution'] = plot2llm.convert(
        fig_dist, 
        format='json',
        detail_level='high',
        include_statistics=True
    )
    
    plt.close(fig_dist)
    return analysis_results
```

### 2. Business Intelligence Dashboard Analysis

```python
import plot2llm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_business_metrics(sales_data, marketing_data):
    """Analyze business KPIs and generate insights"""
    
    # Create comprehensive business dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Business Performance Dashboard', fontsize=16)
    
    # Sales trends
    axes[0, 0].plot(sales_data['date'], sales_data['revenue'], 'b-', linewidth=2)
    axes[0, 0].set_title('Revenue Trends')
    axes[0, 0].set_ylabel('Revenue ($)')
    
    # Customer acquisition cost
    axes[0, 1].scatter(marketing_data['spend'], marketing_data['customers'], alpha=0.7)
    axes[0, 1].set_title('Marketing Efficiency')
    axes[0, 1].set_xlabel('Marketing Spend ($)')
    axes[0, 1].set_ylabel('New Customers')
    
    # Conversion rates
    axes[0, 2].bar(['Website', 'Social', 'Email'], 
                   [sales_data['conversion_web'], 
                    sales_data['conversion_social'], 
                    sales_data['conversion_email']])
    axes[0, 2].set_title('Conversion Rates by Channel')
    
    # Customer lifetime value
    axes[1, 0].hist(sales_data['clv'], bins=15, alpha=0.7, color='green')
    axes[1, 0].set_title('Customer Lifetime Value Distribution')
    
    # Seasonal patterns
    monthly_avg = sales_data.groupby(sales_data['date'].dt.month)['revenue'].mean()
    axes[1, 1].plot(monthly_avg.index, monthly_avg.values, 'r-o')
    axes[1, 1].set_title('Seasonal Revenue Patterns')
    axes[1, 1].set_xlabel('Month')
    
    # ROI analysis
    axes[1, 2].scatter(marketing_data['roi'], marketing_data['channel'], alpha=0.7)
    axes[1, 2].set_title('ROI by Marketing Channel')
    axes[1, 2].set_xlabel('ROI (%)')
    
    plt.tight_layout()
    
    # Generate comprehensive analysis
    dashboard_analysis = plot2llm.convert(
        fig, 
        format='json',
        detail_level='high',
        include_statistics=True,
        include_visual_info=True
    )
    
    plt.close(fig)
    return dashboard_analysis
```

### 3. Scientific Research Data Analysis

```python
import plot2llm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def analyze_experimental_data(treatment_data, control_data, time_points):
    """Analyze scientific experiment results with statistical validation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experimental Results Analysis', fontsize=16)
    
    # Treatment vs Control comparison
    axes[0, 0].plot(time_points, treatment_data, 'b-', label='Treatment', linewidth=2)
    axes[0, 0].plot(time_points, control_data, 'r--', label='Control', linewidth=2)
    axes[0, 0].set_title('Treatment vs Control Over Time')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Response Variable')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Statistical significance test
    t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
    axes[0, 1].bar(['Treatment', 'Control'], 
                   [np.mean(treatment_data), np.mean(control_data)],
                   yerr=[np.std(treatment_data), np.std(control_data)],
                   capsize=5)
    axes[0, 1].set_title(f'Mean Comparison (p={p_value:.4f})')
    axes[0, 1].set_ylabel('Mean Response')
    
    # Residual analysis
    residuals = treatment_data - control_data
    axes[1, 0].scatter(time_points, residuals, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='--')
    axes[1, 0].set_title('Residual Analysis')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Residuals')
    
    # Normality test
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Normality Test of Residuals')
    
    plt.tight_layout()
    
    # Generate scientific analysis
    scientific_analysis = plot2llm.convert(
        fig, 
        format='json',
        detail_level='high',
        include_statistics=True
    )
    
    # Add statistical significance information
    scientific_analysis['statistical_tests'] = {
        't_test': {'statistic': t_stat, 'p_value': p_value},
        'effect_size': np.mean(treatment_data) - np.mean(control_data),
        'significance': p_value < 0.05
    }
    
    plt.close(fig)
    return scientific_analysis
```

### 4. Machine Learning Model Evaluation

```python
import plot2llm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns

def analyze_ml_model_performance(y_true, y_pred, y_proba):
    """Comprehensive ML model evaluation with visual analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Machine Learning Model Performance Analysis', fontsize=16)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[0, 2].plot(recall, precision, 'g-', linewidth=2)
    axes[0, 2].set_title('Precision-Recall Curve')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Prediction Distribution
    axes[1, 0].hist(y_proba[y_true == 0], bins=20, alpha=0.7, label='Negative', color='red')
    axes[1, 0].hist(y_proba[y_true == 1], bins=20, alpha=0.7, label='Positive', color='blue')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Feature Importance (if available)
    feature_importance = np.random.rand(10)  # Example data
    feature_names = [f'Feature_{i}' for i in range(10)]
    axes[1, 1].barh(feature_names, feature_importance)
    axes[1, 1].set_title('Feature Importance')
    axes[1, 1].set_xlabel('Importance Score')
    
    # Model Performance Metrics
    metrics = {
        'Accuracy': 0.85,
        'Precision': 0.82,
        'Recall': 0.88,
        'F1-Score': 0.85
    }
    axes[1, 2].bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
    axes[1, 2].set_title('Model Performance Metrics')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Generate ML analysis
    ml_analysis = plot2llm.convert(
        fig, 
        format='json',
        detail_level='high',
        include_statistics=True
    )
    
    # Add ML-specific metrics
    ml_analysis['ml_metrics'] = metrics
    ml_analysis['model_evaluation'] = {
        'auc_score': 0.92,
        'classification_report': 'Detailed classification report here'
    }
    
    plt.close(fig)
    return ml_analysis
```

### 5. Financial Data Analysis

```python
import plot2llm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_financial_portfolio(portfolio_data, benchmark_data, risk_free_rate=0.02):
    """Analyze financial portfolio performance and risk metrics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Portfolio Performance Analysis', fontsize=16)
    
    # Cumulative returns
    portfolio_cumulative = (1 + portfolio_data['returns']).cumprod()
    benchmark_cumulative = (1 + benchmark_data['returns']).cumprod()
    
    axes[0, 0].plot(portfolio_data.index, portfolio_cumulative, 'b-', label='Portfolio', linewidth=2)
    axes[0, 0].plot(benchmark_data.index, benchmark_cumulative, 'r--', label='Benchmark', linewidth=2)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Risk-return scatter
    portfolio_vol = portfolio_data['returns'].std() * np.sqrt(252)
    portfolio_return = portfolio_data['returns'].mean() * 252
    benchmark_vol = benchmark_data['returns'].std() * np.sqrt(252)
    benchmark_return = benchmark_data['returns'].mean() * 252
    
    axes[0, 1].scatter(portfolio_vol, portfolio_return, s=100, c='blue', alpha=0.7, label='Portfolio')
    axes[0, 1].scatter(benchmark_vol, benchmark_return, s=100, c='red', alpha=0.7, label='Benchmark')
    axes[0, 1].set_title('Risk-Return Analysis')
    axes[0, 1].set_xlabel('Volatility (Annualized)')
    axes[0, 1].set_ylabel('Return (Annualized)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Drawdown analysis
    rolling_max = portfolio_cumulative.expanding().max()
    drawdown = (portfolio_cumulative - rolling_max) / rolling_max
    
    axes[0, 2].fill_between(portfolio_data.index, drawdown, 0, alpha=0.3, color='red')
    axes[0, 2].set_title('Portfolio Drawdown')
    axes[0, 2].set_ylabel('Drawdown')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Returns distribution
    axes[1, 0].hist(portfolio_data['returns'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_title('Returns Distribution')
    axes[1, 0].set_xlabel('Daily Returns')
    axes[1, 0].set_ylabel('Frequency')
    
    # Rolling volatility
    rolling_vol = portfolio_data['returns'].rolling(window=30).std() * np.sqrt(252)
    axes[1, 1].plot(portfolio_data.index, rolling_vol, 'g-', linewidth=2)
    axes[1, 1].set_title('Rolling Volatility (30-day)')
    axes[1, 1].set_ylabel('Volatility')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Sharpe ratio over time
    rolling_sharpe = (portfolio_data['returns'].rolling(window=252).mean() * 252 - risk_free_rate) / \
                     (portfolio_data['returns'].rolling(window=252).std() * np.sqrt(252))
    axes[1, 2].plot(portfolio_data.index, rolling_sharpe, 'purple', linewidth=2)
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('Rolling Sharpe Ratio')
    axes[1, 2].set_ylabel('Sharpe Ratio')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Generate financial analysis
    financial_analysis = plot2llm.convert(
        fig, 
        format='json',
        detail_level='high',
        include_statistics=True
    )
    
    # Add financial metrics
    financial_analysis['financial_metrics'] = {
        'total_return': (portfolio_cumulative.iloc[-1] - 1) * 100,
        'annualized_return': portfolio_return * 100,
        'annualized_volatility': portfolio_vol * 100,
        'sharpe_ratio': (portfolio_return - risk_free_rate) / portfolio_vol,
        'max_drawdown': drawdown.min() * 100,
        'var_95': np.percentile(portfolio_data['returns'], 5) * 100
    }
    
    plt.close(fig)
    return financial_analysis
```

### 6. LLM Integration and Automation

```python
import plot2llm
import openai
import json
from typing import Dict, Any

def create_llm_analysis_prompt(figure_analysis: Dict[str, Any]) -> str:
    """Create a structured prompt for LLM analysis"""
    
    prompt = f"""
    Analyze the following figure data and provide insights:
    
    FIGURE INFORMATION:
    - Type: {figure_analysis.get('figure_type', 'Unknown')}
    - Title: {figure_analysis.get('title', 'No title')}
    - Number of axes: {figure_analysis.get('axes_count', 0)}
    
    CHART TYPES DETECTED:
    {[pt['type'] for ax in figure_analysis.get('axes_info', []) for pt in ax.get('plot_types', [])]}
    
    STATISTICAL SUMMARY:
    {json.dumps(figure_analysis.get('data_info', {}).get('statistics', {}), indent=2)}
    
    Please provide:
    1. Executive summary of the data visualization
    2. Key patterns or trends identified
    3. Statistical insights and their significance
    4. Business or scientific implications
    5. Recommendations based on the data
    6. Potential follow-up analyses needed
    """
    return prompt

def analyze_with_llm(figure, llm_model="gpt-4", api_key=None):
    """Complete analysis pipeline with LLM integration"""
    
    # Step 1: Convert figure to structured format
    figure_analysis = plot2llm.convert(
        figure, 
        format='json',
        detail_level='high',
        include_statistics=True,
        include_visual_info=True
    )
    
    # Step 2: Create LLM prompt
    prompt = create_llm_analysis_prompt(figure_analysis)
    
    # Step 3: Send to LLM (if API key provided)
    if api_key:
        openai.api_key = api_key
        try:
            response = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst and visualization specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            llm_insights = response.choices[0].message.content
        except Exception as e:
            llm_insights = f"LLM analysis failed: {str(e)}"
    else:
        llm_insights = "LLM analysis skipped (no API key provided)"
    
    # Step 4: Combine results
    complete_analysis = {
        'figure_analysis': figure_analysis,
        'llm_insights': llm_insights,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    return complete_analysis

def automated_report_generation(figures_list, report_title="Data Analysis Report"):
    """Generate comprehensive automated reports from multiple figures"""
    
    report = {
        'title': report_title,
        'generated_at': pd.Timestamp.now().isoformat(),
        'total_figures': len(figures_list),
        'figures_analysis': []
    }
    
    for i, fig in enumerate(figures_list):
        try:
            # Analyze each figure
            analysis = plot2llm.convert(
                fig, 
                format='json',
                detail_level='high',
                include_statistics=True
            )
            
            # Add figure metadata
            analysis['figure_id'] = f"figure_{i+1}"
            analysis['analysis_status'] = 'success'
            
            report['figures_analysis'].append(analysis)
            
        except Exception as e:
            # Handle analysis errors
            report['figures_analysis'].append({
                'figure_id': f"figure_{i+1}",
                'analysis_status': 'error',
                'error_message': str(e)
            })
    
    # Generate summary statistics
    successful_analyses = [f for f in report['figures_analysis'] if f['analysis_status'] == 'success']
    report['summary'] = {
        'successful_analyses': len(successful_analyses),
        'failed_analyses': len(report['figures_analysis']) - len(successful_analyses),
        'chart_types_found': list(set([
            pt['type'] for fig in successful_analyses 
            for ax in fig.get('axes_info', []) 
            for pt in ax.get('plot_types', [])
        ]))
    }
    
    return report
```

### 7. Real-Time Monitoring and Alerting

```python
import plot2llm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def monitor_business_metrics(real_time_data, thresholds):
    """Real-time monitoring with automated alerting"""
    
    # Create monitoring dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Real-Time Business Metrics Monitor', fontsize=16)
    
    # Current vs historical comparison
    axes[0, 0].plot(real_time_data['timestamp'], real_time_data['current_metric'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=thresholds['warning'], color='orange', linestyle='--', alpha=0.7, label='Warning')
    axes[0, 0].axhline(y=thresholds['critical'], color='red', linestyle='--', alpha=0.7, label='Critical')
    axes[0, 0].set_title('Real-Time Metric Tracking')
    axes[0, 0].set_ylabel('Metric Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Anomaly detection
    current_value = real_time_data['current_metric'].iloc[-1]
    historical_mean = real_time_data['current_metric'].mean()
    historical_std = real_time_data['current_metric'].std()
    z_score = abs(current_value - historical_mean) / historical_std
    
    axes[0, 1].hist(real_time_data['current_metric'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(x=current_value, color='red', linewidth=2, label=f'Current (z={z_score:.2f})')
    axes[0, 1].set_title('Anomaly Detection')
    axes[0, 1].set_xlabel('Metric Value')
    axes[0, 1].legend()
    
    # Trend analysis
    window = 24  # 24-hour window
    rolling_avg = real_time_data['current_metric'].rolling(window=window).mean()
    axes[1, 0].plot(real_time_data['timestamp'], rolling_avg, 'g-', linewidth=2)
    axes[1, 0].set_title(f'{window}-Hour Rolling Average')
    axes[1, 0].set_ylabel('Rolling Average')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Alert status
    alert_level = 'normal'
    if current_value > thresholds['critical']:
        alert_level = 'critical'
    elif current_value > thresholds['warning']:
        alert_level = 'warning'
    
    colors = {'normal': 'green', 'warning': 'orange', 'critical': 'red'}
    axes[1, 1].bar(['Current Status'], [1], color=colors[alert_level], alpha=0.7)
    axes[1, 1].set_title(f'Alert Status: {alert_level.upper()}')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Analyze monitoring data
    monitoring_analysis = plot2llm.convert(
        fig, 
        format='json',
        detail_level='high',
        include_statistics=True
    )
    
    # Add monitoring-specific information
    monitoring_analysis['monitoring_info'] = {
        'current_value': current_value,
        'alert_level': alert_level,
        'z_score': z_score,
        'thresholds': thresholds,
        'last_updated': datetime.now().isoformat(),
        'trend_direction': 'increasing' if rolling_avg.iloc[-1] > rolling_avg.iloc[-2] else 'decreasing'
    }
    
    plt.close(fig)
    return monitoring_analysis
```

## 🔍 Chart Type Detection

Plot2LLM automatically detects:

- **Scatter Plots** (scatter plots)
- **Line Charts** (line plots)
- **Bar Charts** (bar charts)
- **Histograms** (histograms)
- **Heatmaps** (heatmaps)
- **Box Plots** (box plots)
- **Violin Plots** (violin plots)
- **Density Plots** (density plots)
- **Area Charts** (area plots)
- **Point Plots** (point plots)

## 📈 Statistical Analysis

### Global Statistics
- Mean, median, standard deviation
- Minimum and maximum values
- Outlier detection
- Trend analysis

### Per-Curve Statistics
- Individual analysis of each line/series
- Correlations between variables
- Local variability

### Per-Axis Statistics
- Data distribution by axis
- Skewness and kurtosis
- Ranges and scales

## 🎨 Visual Information

### Colors
- Hexadecimal codes
- Color names
- Used palettes

### Markers
- Marker types
- Marker codes
- Descriptive names

### Line Styles
- Line types (solid, dashed, etc.)
- Line thickness
- Marker styles

## 🚨 Error Handling

```python
import plot2llm

try:
    result = plot2llm.convert(figure)
except ValueError as e:
    print(f"Conversion error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 💡 Best Practices and Tips

### 1. Performance Optimization

```python
# For batch processing, use lower detail levels
for fig in large_figure_list:
    result = plot2llm.convert(fig, detail_level='low')  # Faster processing

# For detailed analysis, use high detail
detailed_result = plot2llm.convert(fig, detail_level='high', include_statistics=True)
```

### 2. Memory Management

```python
# Always close figures after analysis to free memory
fig, ax = plt.subplots()
ax.plot(data)
result = plot2llm.convert(fig)
plt.close(fig)  # Important for memory management

# For multiple figures, process in batches
def process_figures_batch(figures, batch_size=10):
    results = []
    for i in range(0, len(figures), batch_size):
        batch = figures[i:i+batch_size]
        batch_results = []
        for fig in batch:
            result = plot2llm.convert(fig, detail_level='medium')
            batch_results.append(result)
            plt.close(fig)
        results.extend(batch_results)
    return results
```

### 3. Integration with Data Pipelines

```python
def data_analysis_pipeline(dataset, analysis_config):
    """Complete data analysis pipeline with plot2llm integration"""
    
    results = {
        'dataset_info': dataset.info(),
        'figures_analysis': [],
        'summary_insights': {}
    }
    
    # Generate standard visualizations
    figures = generate_standard_plots(dataset, analysis_config)
    
    # Analyze each figure
    for fig_name, fig in figures.items():
        try:
            analysis = plot2llm.convert(
                fig, 
                format='json',
                detail_level='high',
                include_statistics=True
            )
            analysis['figure_name'] = fig_name
            results['figures_analysis'].append(analysis)
            plt.close(fig)
        except Exception as e:
            results['figures_analysis'].append({
                'figure_name': fig_name,
                'error': str(e)
            })
    
    # Generate summary insights
    results['summary_insights'] = generate_summary_insights(results['figures_analysis'])
    
    return results
```

### 4. Custom Analysis Extensions

```python
from plot2llm.analyzers import MatplotlibAnalyzer

class BusinessAnalyzer(MatplotlibAnalyzer):
    """Custom analyzer for business-specific metrics"""
    
    def analyze(self, figure, **kwargs):
        # Get base analysis
        analysis = super().analyze(figure, **kwargs)
        
        # Add business-specific metrics
        analysis['business_metrics'] = self.extract_business_metrics(figure)
        
        return analysis
    
    def extract_business_metrics(self, figure):
        """Extract business-relevant metrics from figure"""
        metrics = {}
        
        # Example: Extract trend information
        if hasattr(figure, 'axes'):
            for ax in figure.axes:
                if ax.lines:
                    # Calculate trend direction
                    for line in ax.lines:
                        y_data = line.get_ydata()
                        if len(y_data) > 1:
                            trend = 'increasing' if y_data[-1] > y_data[0] else 'decreasing'
                            metrics['trend_direction'] = trend
                            metrics['trend_magnitude'] = abs(y_data[-1] - y_data[0])
        
        return metrics
```

### 5. Quality Assurance

```python
def validate_analysis_quality(figure_analysis):
    """Validate the quality and completeness of analysis results"""
    
    quality_score = 0
    issues = []
    
    # Check required fields
    required_fields = ['figure_type', 'axes_info', 'data_info']
    for field in required_fields:
        if field not in figure_analysis:
            issues.append(f"Missing required field: {field}")
        else:
            quality_score += 1
    
    # Check data completeness
    if 'data_info' in figure_analysis:
        data_info = figure_analysis['data_info']
        if 'statistics' in data_info and data_info['statistics']:
            quality_score += 1
        else:
            issues.append("Missing statistical information")
    
    # Check visual information
    if 'visual_info' in figure_analysis:
        quality_score += 1
    
    # Normalize score to 0-100
    quality_score = (quality_score / 4) * 100
    
    return {
        'quality_score': quality_score,
        'issues': issues,
        'is_acceptable': quality_score >= 75
    }
```

## 🔧 Advanced Configuration

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

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Main dependencies**: numpy, pandas
- **Optional dependencies**: matplotlib, seaborn, plotly

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes and commit (`git commit -am 'Add new feature'`)
4. Push to your branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- matplotlib, seaborn, and plotly communities
- Library contributors and users
- Python and machine learning community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/plot2llm/issues)
- **Documentation**: [Complete Documentation](https://plot2llm.readthedocs.io)
- **Email**: your-email@example.com

---

**Plot2LLM** - Convert visualizations into LLM insights 🚀