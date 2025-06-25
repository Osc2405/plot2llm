#!/usr/bin/env python3
"""
Advanced example demonstrating rich data analysis capabilities for LLMs.

This script creates various types of plots and shows how plot2llm can extract
detailed information that allows LLMs to understand and analyze the data
presented in visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot2llm import FigureConverter


def create_trend_analysis_example():
    """Create an example showing trend analysis capabilities."""
    print("📈 Trend Analysis Example")
    print("=" * 50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear trend with seasonal component
    x = np.linspace(0, 24, 100)
    linear_trend = 2 * x + 10
    seasonal = 5 * np.sin(2 * np.pi * x / 12)
    noise = np.random.normal(0, 1, 100)
    y1 = linear_trend + seasonal + noise
    
    ax1.plot(x, y1, 'b-', linewidth=2, label='Data with Trend + Seasonality')
    ax1.plot(x, linear_trend, 'r--', linewidth=2, label='Linear Trend')
    ax1.set_title('Time Series with Linear Trend and Seasonality')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Exponential growth
    x2 = np.linspace(0, 10, 100)
    y2 = 10 * np.exp(0.3 * x2) + np.random.normal(0, 5, 100)
    
    ax2.plot(x2, y2, 'g-', linewidth=2, label='Exponential Growth')
    ax2.set_title('Exponential Growth Pattern')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Analyze with plot2llm
    converter = FigureConverter()
    
    print("\n🔍 Analysis Results:")
    print("-" * 30)
    
    # Text analysis
    text_result = converter.convert(fig, output_format='text')
    print("Text Analysis (first 500 chars):")
    print(text_result[:500] + "...")
    
    # Semantic analysis
    semantic_result = converter.convert(fig, output_format='semantic')
    print(f"\n📊 Semantic Analysis Keys: {list(semantic_result.keys())}")
    
    # Extract specific insights
    if 'data_info' in semantic_result and 'statistics' in semantic_result['data_info']:
        stats = semantic_result['data_info']['statistics']
        print(f"\n📈 Statistical Summary:")
        print(f"  - Mean: {stats.get('mean', 'N/A')}")
        print(f"  - Standard Deviation: {stats.get('std', 'N/A')}")
        print(f"  - Range: {stats.get('min', 'N/A')} to {stats.get('max', 'N/A')}")
    
    plt.close(fig)
    return semantic_result


def create_correlation_analysis_example():
    """Create an example showing correlation analysis."""
    print("\n\n🔗 Correlation Analysis Example")
    print("=" * 50)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    np.random.seed(42)
    
    # Strong positive correlation
    x1 = np.random.normal(0, 1, 100)
    y1 = 2 * x1 + np.random.normal(0, 0.3, 100)
    ax1.scatter(x1, y1, alpha=0.6, color='blue')
    ax1.set_title('Strong Positive Correlation (r ≈ 0.95)')
    ax1.set_xlabel('Variable X')
    ax1.set_ylabel('Variable Y')
    
    # Strong negative correlation
    x2 = np.random.normal(0, 1, 100)
    y2 = -2 * x2 + np.random.normal(0, 0.3, 100)
    ax2.scatter(x2, y2, alpha=0.6, color='red')
    ax2.set_title('Strong Negative Correlation (r ≈ -0.95)')
    ax2.set_xlabel('Variable X')
    ax2.set_ylabel('Variable Y')
    
    # Weak correlation
    x3 = np.random.normal(0, 1, 100)
    y3 = 0.2 * x3 + np.random.normal(0, 1, 100)
    ax3.scatter(x3, y3, alpha=0.6, color='green')
    ax3.set_title('Weak Positive Correlation (r ≈ 0.2)')
    ax3.set_xlabel('Variable X')
    ax3.set_ylabel('Variable Y')
    
    # No correlation
    x4 = np.random.normal(0, 1, 100)
    y4 = np.random.normal(0, 1, 100)
    ax4.scatter(x4, y4, alpha=0.6, color='orange')
    ax4.set_title('No Correlation (r ≈ 0)')
    ax4.set_xlabel('Variable X')
    ax4.set_ylabel('Variable Y')
    
    plt.tight_layout()
    
    # Analyze with plot2llm
    converter = FigureConverter()
    
    print("\n🔍 Correlation Analysis Results:")
    print("-" * 40)
    
    text_result = converter.convert(fig, output_format='text')
    print("Text Analysis (first 400 chars):")
    print(text_result[:400] + "...")
    
    semantic_result = converter.convert(fig, output_format='semantic')
    
    # Extract correlation insights
    if 'axes_info' in semantic_result:
        print(f"\n📊 Correlation Patterns Detected:")
        for i, axis in enumerate(semantic_result['axes_info']):
            title = axis.get('title', f'Subplot {i+1}')
            print(f"  - {title}")
    
    plt.close(fig)
    return semantic_result


def create_distribution_analysis_example():
    """Create an example showing distribution analysis."""
    print("\n\n📊 Distribution Analysis Example")
    print("=" * 50)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    np.random.seed(42)
    
    # Normal distribution
    normal_data = np.random.normal(0, 1, 1000)
    ax1.hist(normal_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Normal Distribution (μ=0, σ=1)')
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Frequency')
    ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='Mean')
    ax1.legend()
    
    # Skewed distribution
    skewed_data = np.random.exponential(2, 1000)
    ax2.hist(skewed_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Right-Skewed Distribution (Exponential)')
    ax2.set_xlabel('Values')
    ax2.set_ylabel('Frequency')
    
    # Bimodal distribution
    bimodal_data = np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])
    ax3.hist(bimodal_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Bimodal Distribution')
    ax3.set_xlabel('Values')
    ax3.set_ylabel('Frequency')
    
    # Uniform distribution
    uniform_data = np.random.uniform(-3, 3, 1000)
    ax4.hist(uniform_data, bins=30, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_title('Uniform Distribution')
    ax4.set_xlabel('Values')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Analyze with plot2llm
    converter = FigureConverter()
    
    print("\n🔍 Distribution Analysis Results:")
    print("-" * 40)
    
    text_result = converter.convert(fig, output_format='text')
    print("Text Analysis (first 400 chars):")
    print(text_result[:400] + "...")
    
    semantic_result = converter.convert(fig, output_format='semantic')
    
    # Extract distribution insights
    if 'data_info' in semantic_result and 'statistics' in semantic_result['data_info']:
        stats = semantic_result['data_info']['statistics']
        print(f"\n📈 Distribution Statistics:")
        print(f"  - Data points: {semantic_result['data_info'].get('data_points', 'N/A')}")
        print(f"  - Mean: {stats.get('mean', 'N/A'):.3f}")
        print(f"  - Standard Deviation: {stats.get('std', 'N/A'):.3f}")
        print(f"  - Range: {stats.get('min', 'N/A'):.3f} to {stats.get('max', 'N/A'):.3f}")
    
    plt.close(fig)
    return semantic_result


def create_business_insights_example():
    """Create an example showing business insights extraction."""
    print("\n\n💼 Business Insights Example")
    print("=" * 50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sales data over time
    years = np.arange(2018, 2024)
    sales = [100, 120, 150, 180, 220, 280]
    costs = [80, 95, 115, 140, 170, 200]
    profits = [sales[i] - costs[i] for i in range(len(sales))]
    
    # Sales trend
    ax1.plot(years, sales, 'bo-', linewidth=3, markersize=8, label='Sales')
    ax1.plot(years, costs, 'ro-', linewidth=3, markersize=8, label='Costs')
    ax1.plot(years, profits, 'go-', linewidth=3, markersize=8, label='Profits')
    ax1.set_title('Company Performance (2018-2023)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Amount (in thousands)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Market share by product
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    market_share = [35, 25, 20, 20]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    wedges, texts, autotexts = ax2.pie(market_share, labels=products, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax2.set_title('Market Share by Product')
    
    plt.tight_layout()
    
    # Analyze with plot2llm
    converter = FigureConverter()
    
    print("\n🔍 Business Insights Results:")
    print("-" * 40)
    
    text_result = converter.convert(fig, output_format='text')
    print("Text Analysis (first 500 chars):")
    print(text_result[:500] + "...")
    
    semantic_result = converter.convert(fig, output_format='semantic')
    
    # Extract business insights
    if 'basic_info' in semantic_result:
        title = semantic_result['basic_info'].get('title', '')
        print(f"\n📊 Business Analysis:")
        print(f"  - Chart Title: {title}")
        print(f"  - Chart Type: {semantic_result['basic_info'].get('figure_type', 'N/A')}")
        print(f"  - Number of Subplots: {len(semantic_result.get('axes_info', []))}")
    
    if 'data_info' in semantic_result and 'statistics' in semantic_result['data_info']:
        stats = semantic_result['data_info']['statistics']
        print(f"  - Data Statistics Available: {list(stats.keys())}")
    
    plt.close(fig)
    return semantic_result


def main():
    """Main function to run all advanced analysis examples."""
    print("🚀 Plot2LLM Advanced Data Analysis Examples")
    print("=" * 60)
    print("This example demonstrates how plot2llm can extract rich information")
    print("that allows LLMs to understand and analyze data visualizations.")
    print("=" * 60)
    
    # Run all examples
    results = {}
    
    try:
        results['trend'] = create_trend_analysis_example()
        results['correlation'] = create_correlation_analysis_example()
        results['distribution'] = create_distribution_analysis_example()
        results['business'] = create_business_insights_example()
        
        print("\n\n✅ All Examples Completed Successfully!")
        print("=" * 60)
        print("📋 Summary of Analysis Types:")
        print("  - Trend Analysis: Linear and exponential patterns")
        print("  - Correlation Analysis: Various correlation strengths")
        print("  - Distribution Analysis: Different data distributions")
        print("  - Business Insights: Performance and market data")
        
        print("\n🎯 Key Benefits for LLMs:")
        print("  - Detailed statistical summaries")
        print("  - Pattern recognition capabilities")
        print("  - Business context extraction")
        print("  - Multi-format output (text, JSON, semantic)")
        print("  - Rich metadata for analysis")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 