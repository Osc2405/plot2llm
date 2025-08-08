# 🧠 Plot2LLM v0.2.1: LLM Analysis Guide

This guide explains how to use `plot2llm` v0.2.0 to extract rich information that allows LLMs (Large Language Models) to understand and analyze data visualizations effectively. The new version includes complete statistical analysis, improved plot type detection, and advanced analysis capabilities.

## 📋 Index

1. [Introduction](#introduction)
2. [Analysis Capabilities v0.2.0](#analysis-capabilities-v020)
3. [Output Formats](#output-formats)
4. [Use Cases for LLMs](#use-cases-for-llms)
5. [Practical Examples](#practical-examples)
6. [Best Practices](#best-practices)
7. [LLM Integration](#llm-integration)
8. [Advanced Statistical Analysis](#advanced-statistical-analysis)

## 🎯 Introduction

`plot2llm` v0.2.0 is a Python library that converts figures and visualizations into formats that LLMs can easily understand and process. This version includes complete statistical analysis and improved capabilities that allow LLMs to:

- **Analyze patterns** in visualized data with complete statistical analysis
- **Extract insights** statistically and business-wise automatically
- **Generate reports** based on visualizations with statistical evidence
- **Answer questions** about charts and data with deep analysis
- **Compare multiple** visualizations with statistical metrics
- **Detect outliers** and anomalies automatically
- **Analyze distributions** with skewness and kurtosis

## 🔍 Analysis Capabilities v0.2.0

### 1. **Complete Statistical Analysis**
- **Central Tendency**: mean, median, mode
- **Variability Analysis**: std, variance, range
- **Distribution Analysis**: skewness, kurtosis with interpretations
- **Correlation Analysis**: Pearson with strength and direction
- **Outlier Detection**: IQR method for all plot types
- **Data Quality Assessment**: total points, missing values

### 2. **Trend Analysis**
- Detection of linear, exponential, and seasonal patterns
- Inflection point identification
- Growth and decline analysis
- Monotonicity detection (increasing, decreasing, stable)

### 3. **Correlation Analysis**
- Detection of positive and negative correlations
- Identification of dispersion patterns
- Analysis of relationship strength (weak, moderate, strong)
- Correlation direction (positive, negative, none)

### 4. **Distribution Analysis**
- Identification of normal, skewed, bimodal, multimodal distributions
- Detection of outliers and atypical values
- Analysis of data shape and characteristics
- Bin analysis for histograms

### 5. **Business Analysis**
- Business context extraction
- Key metrics identification
- Performance analysis and comparisons
- Unified LLM Description and Context

## 📊 Output Formats

### 1. **Text Format** (`text`)
```python
from plot2llm import FigureConverter

converter = FigureConverter()
text_result = converter.convert(fig, format='text')
```

**Advantages for LLMs:**
- Easy to process and understand
- Structured information in natural language
- Includes statistics and metadata

**Output example:**
```
Figure type: matplotlib.figure
Dimensions (inches): [10. 6.]
Title: Sales Growth Analysis
Number of axes: 1

Axis 0: type=linear, xlabel=Year, ylabel=Sales, 
x_range=(2018, 2023), y_range=(100, 300), grid=True, legend=True

Data points: 6
Data types: ['line_plot']
Statistics: mean=200.0, std=71.4, min=100, max=300, median=175.0

Colors: ['#1f77b4']
Markers: [<matplotlib.markers.MarkerStyle object>]
Line styles: ['-']
Background color: #ffffff
```

### 2. **JSON Format** (`json`)
```python
json_result = converter.convert(fig, format='json')
```

**Advantages for LLMs:**
- Clear and parseable data structure
- Easy extraction of specific fields
- Compatible with APIs and systems

### 3. **Semantic Format** (`semantic`) - Enhanced v0.2.0
```python
semantic_result = converter.convert(fig, format='semantic', include_statistics=True)
```

**Advantages for LLMs:**
- Richer and more contextual information
- Additional metadata
- Analysis-optimized structure
- Complete statistical analysis included
- Unified LLM Description and Context

**Enhanced output example:**
```json
{
  "metadata": {
    "figure_type": "matplotlib.figure",
    "dimensions": "[10. 6.]",
    "title": "Sales Growth Analysis",
    "axes_count": 1
  },
  "axes": [
    {
      "plot_type": "line",
      "xlabel": "Year",
      "ylabel": "Sales",
      "x_range": [2018, 2023],
      "y_range": [100, 300],
      "has_grid": true,
      "has_legend": true,
      "stats": {
        "central_tendency": {
          "mean": 200.0,
          "median": 175.0,
          "mode": null
        },
        "variability": {
          "standard_deviation": 71.4,
          "variance": 5097.96,
          "range": {"min": 100, "max": 300}
        }
      }
    }
  ],
  "statistical_insights": {
    "central_tendency": {
      "mean": 200.0,
      "median": 175.0,
      "mode": null
    },
    "variability": {
      "standard_deviation": 71.4,
      "variance": 5097.96,
      "range": {"min": 100, "max": 300}
    },
    "distribution": {
      "skewness": 0.5,
      "kurtosis": -0.8,
      "skewness_interpretation": "right_skewed",
      "kurtosis_interpretation": "platykurtic"
    },
    "outliers": {
      "detected": true,
      "count": 2,
      "x_outliers": 0,
      "y_outliers": 2
    },
    "correlations": [
      {
        "type": "pearson",
        "value": 0.95,
        "strength": "strong",
        "direction": "positive"
      }
    ]
  },
  "llm_description": {
    "one_sentence_summary": "This visualization shows a linear_trend relationship.",
    "structured_analysis": ["what", "when", "why", "how"],
    "key_insights": [
      {
        "type": "pattern",
        "description": "Strong positive linear trend",
        "confidence": 0.90
      }
    ]
  },
  "llm_context": {
    "interpretation_hints": [
      {
        "type": "trend_analysis",
        "description": "Look for trends, slopes, and inflection points.",
        "priority": "high",
        "category": "line_plot"
      }
    ],
    "analysis_suggestions": ["Consider seasonality", "Check for outliers"],
    "common_questions": ["What is the growth rate?", "Are there any anomalies?"],
    "related_concepts": ["time_series", "trend_analysis", "growth_metrics"]
  }
}
```

## 🎯 Use Cases for LLMs

### 1. **Automatic Report Analysis with Statistics**
```python
# The LLM can automatically analyze multiple charts with statistical analysis
def analyze_report_charts(charts):
    insights = []
    for chart in charts:
        result = converter.convert(chart, format='semantic', include_statistics=True)
        insights.append(extract_statistical_insights(result))
    return generate_statistical_report(insights)
```

### 2. **Answering Questions about Data with Statistical Evidence**
```python
# The LLM can answer specific questions about visualizations with statistical analysis
def answer_chart_question(chart, question):
    analysis = converter.convert(chart, format='semantic', include_statistics=True)
    return llm.answer(f"Based on this chart analysis with statistical evidence: {analysis}\nQuestion: {question}")
```

### 3. **Comparing Visualizations with Statistical Metrics**
```python
# The LLM can compare multiple charts using statistical analysis
def compare_charts_statistically(chart1, chart2):
    analysis1 = converter.convert(chart1, format='semantic', include_statistics=True)
    analysis2 = converter.convert(chart2, format='semantic', include_statistics=True)
    return llm.compare_statistical_analysis(analysis1, analysis2)
```

### 4. **Generating Business Insights with Statistical Analysis**
```python
# The LLM can extract business insights with statistical evidence
def extract_business_insights_with_stats(chart):
    analysis = converter.convert(chart, format='semantic', include_statistics=True)
    return llm.extract_statistical_insights(analysis, context="business_analysis")
```

## 💡 Practical Examples

### Example 1: Sales Trend Analysis with Statistics
```python
import matplotlib.pyplot as plt
import numpy as np
from plot2llm import FigureConverter

# Create sales chart
fig, ax = plt.subplots()
years = [2019, 2020, 2021, 2022, 2023]
sales = [100, 120, 150, 200, 280]
ax.plot(years, sales, 'bo-', linewidth=2)
ax.set_title('Company Sales Growth')
ax.set_xlabel('Year')
ax.set_ylabel('Sales (in thousands)')

# Analyze for LLM with complete statistical analysis
converter = FigureConverter()
analysis = converter.convert(fig, format='semantic', include_statistics=True)

# The LLM can now analyze:
# - Annual growth rate with statistical evidence
# - Seasonal patterns with distribution analysis
# - Future projections based on trends
# - Business insights with outliers and correlations
# - Skewness and kurtosis analysis to understand distribution
```

### Example 2: Correlation Analysis with Statistical Metrics
```python
# Create correlation chart
fig, ax = plt.subplots()
x = np.random.normal(0, 1, 100)
y = 2 * x + np.random.normal(0, 0.5, 100)
ax.scatter(x, y, alpha=0.6)
ax.set_title('Correlation Analysis')

# Analyze for LLM with complete statistical analysis
analysis = converter.convert(fig, format='semantic', include_statistics=True)

# The LLM can identify:
# - Correlation strength with numerical value
# - Relationship direction (positive/negative)
# - Statistical significance
# - Practical implications
# - Outlier detection in correlation
# - Data distribution analysis
```

### Example 3: Distribution Analysis with Histograms
```python
# Create histogram with distribution analysis
fig, ax = plt.subplots()
data = np.random.normal(0, 1, 1000)
ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
ax.set_title('Distribution Analysis')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')

# Analyze for LLM with complete statistical analysis
analysis = converter.convert(fig, format='semantic', include_statistics=True)

# The LLM can analyze:
# - Distribution type (normal, skewed, bimodal)
# - Skewness and kurtosis with interpretations
# - Outlier detection
# - Bin and frequency analysis
# - Comparison with theoretical distributions
```

## 🚀 Best Practices v0.2.0

### 1. **Optimal Configuration for LLMs with Statistical Analysis**
```python
# Use high detail level for complete analysis with statistics
converter = FigureConverter()
result = converter.convert(
    fig, 
    format='semantic',
    detail_level='high',
    include_statistics=True,
    include_data=True,
    include_curve_points=True  # For detailed analysis
)
```

### 2. **Processing Multiple Charts with Statistical Analysis**
```python
def batch_analyze_charts_with_stats(charts):
    """Analyzes multiple charts efficiently with statistical analysis."""
    converter = FigureConverter()
    results = []
    
    for chart in charts:
        try:
            result = converter.convert(chart, format='semantic', include_statistics=True)
            results.append(result)
        except Exception as e:
            results.append({'error': str(e)})
    
    return results
```

### 3. **Extracting Specific Statistical Insights**
```python
def extract_statistical_insights(analysis):
    """Extracts specific statistical insights."""
    if 'statistical_insights' in analysis:
        stats = analysis['statistical_insights']
        
        insights = {
            'central_tendency': stats.get('central_tendency', {}),
            'variability': stats.get('variability', {}),
            'distribution': stats.get('distribution', {}),
            'outliers': stats.get('outliers', {}),
            'correlations': stats.get('correlations', [])
        }
        
        return insights
    return None
```

## 🔗 LLM Integration

### 1. **Prompt Engineering for Statistical Analysis**
```python
def create_statistical_analysis_prompt(chart_analysis):
    return f"""
    Analyze the following data visualization with complete statistical analysis:
    
    {chart_analysis}
    
    Please provide:
    1. An executive summary of the data with statistical evidence
    2. Main patterns identified with metrics
    3. Relevant business insights with statistical analysis
    4. Recommendations based on data and statistics
    5. Important limitations or considerations
    6. Outlier and anomaly analysis if any
    7. Interpretation of correlations and distributions
    """
```

### 2. **Integration with LLM APIs for Statistical Analysis**
```python
import openai

def analyze_with_gpt4_statistical(chart):
    converter = FigureConverter()
    analysis = converter.convert(chart, format='semantic', include_statistics=True)
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert data analyst with advanced statistical knowledge."},
            {"role": "user", "content": create_statistical_analysis_prompt(analysis)}
        ]
    )
    
    return response.choices[0].message.content
```

### 3. **Comparative Analysis with Statistical Metrics**
```python
def compare_multiple_charts_statistically(charts, comparison_criteria):
    converter = FigureConverter()
    analyses = []
    
    for chart in charts:
        analysis = converter.convert(chart, format='semantic', include_statistics=True)
        analyses.append(analysis)
    
    # The LLM can compare multiple statistical analyses
    comparison_prompt = f"""
    Compare the following visualizations statistically based on: {comparison_criteria}
    
    Analysis 1: {analyses[0]}
    Analysis 2: {analyses[1]}
    ...
    
    Provide a structured comparison with:
    - Central tendency comparison
    - Variability analysis
    - Distribution comparison
    - Outlier analysis
    - Correlation comparison
    """
    
    return llm.analyze(comparison_prompt)
```

## 📈 Advanced Statistical Analysis

### 1. **Central Tendency Analysis**
```python
def analyze_central_tendency(analysis):
    """Analyzes central tendency measures."""
    if 'statistical_insights' in analysis:
        central_tendency = analysis['statistical_insights'].get('central_tendency', {})
        
        return {
            'mean': central_tendency.get('mean'),
            'median': central_tendency.get('median'),
            'mode': central_tendency.get('mode'),
            'interpretation': interpret_central_tendency(central_tendency)
        }
    return None
```

### 2. **Variability Analysis**
```python
def analyze_variability(analysis):
    """Analyzes variability measures."""
    if 'statistical_insights' in analysis:
        variability = analysis['statistical_insights'].get('variability', {})
        
        return {
            'standard_deviation': variability.get('standard_deviation'),
            'variance': variability.get('variance'),
            'range': variability.get('range', {}),
            'interpretation': interpret_variability(variability)
        }
    return None
```

### 3. **Distribution Analysis**
```python
def analyze_distribution(analysis):
    """Analyzes data distribution."""
    if 'statistical_insights' in analysis:
        distribution = analysis['statistical_insights'].get('distribution', {})
        
        return {
            'skewness': distribution.get('skewness'),
            'kurtosis': distribution.get('kurtosis'),
            'skewness_interpretation': distribution.get('skewness_interpretation'),
            'kurtosis_interpretation': distribution.get('kurtosis_interpretation'),
            'shape_analysis': interpret_distribution_shape(distribution)
        }
    return None
```

### 4. **Outlier Detection**
```python
def analyze_outliers(analysis):
    """Analyzes detected outliers."""
    if 'statistical_insights' in analysis:
        outliers = analysis['statistical_insights'].get('outliers', {})
        
        return {
            'detected': outliers.get('detected', False),
            'count': outliers.get('count', 0),
            'x_outliers': outliers.get('x_outliers', 0),
            'y_outliers': outliers.get('y_outliers', 0),
            'interpretation': interpret_outliers(outliers)
        }
    return None
```

### 5. **Correlation Analysis**
```python
def analyze_correlations(analysis):
    """Analyzes detected correlations."""
    if 'statistical_insights' in analysis:
        correlations = analysis['statistical_insights'].get('correlations', [])
        
        return [{
            'type': corr.get('type'),
            'value': corr.get('value'),
            'strength': corr.get('strength'),
            'direction': corr.get('direction'),
            'interpretation': interpret_correlation(corr)
        } for corr in correlations]
    return []
```

## 📈 Quality Metrics v0.2.0

### Statistical Analysis Quality Indicators:
- **Statistical Completeness**: Is all relevant statistical information extracted?
- **Statistical Accuracy**: Are the extracted statistical data correct?
- **Statistical Context**: Is the appropriate statistical context captured?
- **Statistical Structure**: Is the statistical information well organized?
- **Statistical Interpretation**: Are useful interpretations provided?

### Statistical Performance Evaluation:
```python
def evaluate_statistical_analysis_quality(original_chart, llm_analysis):
    """Evaluates the quality of statistical analysis performed by the LLM."""
    converter = FigureConverter()
    ground_truth = converter.convert(original_chart, format='semantic', include_statistics=True)
    
    # Compare with LLM analysis
    statistical_accuracy = compare_statistical_analyses(ground_truth, llm_analysis)
    statistical_completeness = check_statistical_completeness(llm_analysis)
    statistical_relevance = assess_statistical_relevance(llm_analysis)
    
    return {
        'statistical_accuracy': statistical_accuracy,
        'statistical_completeness': statistical_completeness,
        'statistical_relevance': statistical_relevance,
        'overall_statistical_score': (statistical_accuracy + statistical_completeness + statistical_relevance) / 3
    }
```

## 🎯 Conclusion v0.2.0

`plot2llm` v0.2.0 provides a solid foundation for LLMs to analyze data visualizations effectively with complete statistical analysis. By combining structured information extraction with advanced statistical analysis and LLM natural language processing capabilities, valuable insights and deep analysis of visual data can be obtained.

### New v0.2.0 Capabilities:
1. **Complete statistical analysis** for all plot types
2. **Automatic outlier detection**
3. **Distribution analysis** with skewness and kurtosis
4. **Correlations** with strength and direction
5. **Unified LLM Description and Context**
6. **Expanded test suite** (172/174 tests passing)
7. **Optimized performance** (24s vs 57s)

### Next Steps:
1. **Experiment** with different visualization types and statistical analysis
2. **Refine prompts** for specific use cases with statistical evidence
3. **Integrate** with existing data analysis systems
4. **Develop** additional capabilities based on specific needs

---

*For more information, consult the complete documentation of `plot2llm` v0.2.0 and the examples included in the repository.* 