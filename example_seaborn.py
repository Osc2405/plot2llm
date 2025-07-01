"""
Example demonstrating seaborn support in plot2llm.

This example shows how to use the library with various seaborn plot types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot2llm import FigureConverter, TextFormatter, JSONFormatter
import json
import os

def create_sample_data():
    """Create sample data for demonstrations."""
    np.random.seed(42)
    
    # Create a sample dataset
    data = {
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'size': np.random.uniform(10, 100, 100),
        'value': np.random.uniform(0, 1, 100)
    }
    
    return pd.DataFrame(data)

def print_llm_analysis(semantic_result, fig_name):
    print(f"\n📊 Semantic Analysis Keys: {list(semantic_result.keys())}")
    print(f"  - Figure type: {semantic_result['figure_info']['figure_type']}")
    
    # Colors information
    colors = semantic_result['colors']
    print(f"  - Colors: {[f'{c['hex']} ({c['name']})' if c['name'] else c['hex'] for c in colors]}")
    
    # Statistics information
    stats = semantic_result['statistics']
    if 'global' in stats:
        g = stats['global']
        print(f"  - Global stats: mean={g['mean']}, std={g['std']}, min={g['min']}, max={g['max']}, median={g['median']}")
    
    if 'per_curve' in stats:
        for i, curve in enumerate(stats['per_curve']):
            # Handle both old format (mean) and new format (x_mean, y_mean)
            if 'mean' in curve:
                mean_val = curve['mean']
                trend_val = curve.get('trend', 0)
            elif 'x_mean' in curve and 'y_mean' in curve:
                mean_val = f"x={curve['x_mean']:.3f}, y={curve['y_mean']:.3f}"
                trend_val = curve.get('trend', 0)
            else:
                mean_val = "N/A"
                trend_val = 0
            
            outliers_count = len(curve.get('outliers', []))
            print(f"    Curve {i+1} ({curve['label']}): mean={mean_val}, trend={trend_val}, outliers={outliers_count}")
    
    if 'per_axis' in stats:
        print(f"  - Statistics per axis/subplot:")
        for axis in stats['per_axis']:
            title = axis.get('title', f'Subplot {axis.get("axis_index")+1}')
            if axis.get('mean') is not None:
                print(f"    Axis {axis.get('axis_index')} ({title}): mean={axis.get('mean')}, std={axis.get('std')}, skewness={axis.get('skewness')}, kurtosis={axis.get('kurtosis')}, outliers={len(axis.get('outliers', []))}")
            else:
                print(f"    Axis {axis.get('axis_index')} ({title}): no data")
    
    # Axis information
    axis_info = semantic_result['axis_info']
    print(f"  - Total axes: {axis_info.get('total_axes', 0)}")
    print(f"  - Figure title: {axis_info.get('figure_title', 'Not found')}")
    
    # Guardar resultados
    if not os.path.exists('examples_seaborn'):
        os.makedirs('examples_seaborn')
    if fig_name is not None:
        plt.savefig(f'examples_seaborn/{fig_name}.png')
        with open(f'examples_seaborn/{fig_name}.json', 'w') as f:
            json.dump(semantic_result, f, default=str)

def example_seaborn_scatter():
    """Example with seaborn scatter plot."""
    print("=== Seaborn Scatter Plot Example ===")
    
    # Create data
    df = create_sample_data()
    
    # Create seaborn scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='category', size='size')
    plt.title('Seaborn Scatter Plot Example')
    
    # Get the figure
    fig = plt.gcf()
    
    # Convert to LLM format
    converter = FigureConverter()
    semantic_result = converter.convert(fig, output_format="semantic")
    print_llm_analysis(semantic_result, 'scatter')
    plt.close(fig)
    print("\n" + "="*50 + "\n")

def example_seaborn_heatmap():
    """Example with seaborn heatmap."""
    print("=== Seaborn Heatmap Example ===")
    
    # Create correlation matrix
    df = create_sample_data()
    corr_matrix = df[['x', 'y', 'size', 'value']].corr()
    
    # Create seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    # Get the figure
    fig = plt.gcf()
    
    # Convert to LLM format
    converter = FigureConverter()
    semantic_result = converter.convert(fig, output_format="semantic")
    print_llm_analysis(semantic_result, 'heatmap')
    plt.close(fig)
    print("\n" + "="*50 + "\n")

def example_seaborn_facetgrid():
    """Example with seaborn FacetGrid."""
    print("=== Seaborn FacetGrid Example ===")
    
    # Create data
    df = create_sample_data()
    
    # Create seaborn FacetGrid
    g = sns.FacetGrid(df, col="category", height=4, aspect=1.2)
    g.map_dataframe(sns.scatterplot, x="x", y="y", size="size")
    g.fig.suptitle('FacetGrid Example', y=1.02)
    g.fig.tight_layout()
    
    # Convert to LLM format
    converter = FigureConverter()
    semantic_result = converter.convert(g, output_format="semantic")
    print_llm_analysis(semantic_result, 'facetgrid')
    plt.close(g.fig)
    print("\n" + "="*50 + "\n")

def example_seaborn_distribution():
    """Example with seaborn distribution plots."""
    print("=== Seaborn Distribution Plots Example ===")
    
    # Create data
    df = create_sample_data()
    
    # Create subplots with different distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    sns.histplot(data=df, x='x', ax=axes[0, 0])
    axes[0, 0].set_title('Histogram')
    
    # KDE plot
    sns.kdeplot(data=df, x='y', ax=axes[0, 1])
    axes[0, 1].set_title('KDE Plot')
    
    # Box plot
    sns.boxplot(data=df, x='category', y='value', ax=axes[1, 0])
    axes[1, 0].set_title('Box Plot')
    
    # Violin plot
    sns.violinplot(data=df, x='category', y='size', ax=axes[1, 1])
    axes[1, 1].set_title('Violin Plot')
    
    plt.tight_layout()
    plt.suptitle('Distribution Plots', y=1.02)
    
    # Convert to LLM format
    converter = FigureConverter()
    semantic_result = converter.convert(fig, output_format="semantic")
    print_llm_analysis(semantic_result, 'distribution')
    plt.close(fig)
    print("\n" + "="*50 + "\n")

def example_seaborn_regression():
    """Example with seaborn regression plots."""
    print("=== Seaborn Regression Plots Example ===")
    
    # Create data
    df = create_sample_data()
    
    # Create seaborn regression plot
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='x', y='y', scatter_kws={'alpha': 0.6})
    plt.title('Regression Plot Example')
    
    # Get the figure
    fig = plt.gcf()
    
    # Convert to LLM format
    converter = FigureConverter()
    semantic_result = converter.convert(fig, output_format="semantic")
    print_llm_analysis(semantic_result, 'regression')
    plt.close(fig)
    print("\n" + "="*50 + "\n")

def example_seaborn_pairplot():
    """Example with seaborn pairplot."""
    print("=== Seaborn Pairplot Example ===")
    
    # Create data
    df = create_sample_data()
    
    # Create seaborn pairplot
    pair_plot = sns.pairplot(df[['x', 'y', 'size', 'value']], diag_kind='kde')
    pair_plot.fig.suptitle('Pairplot Example', y=1.02)
    pair_plot.fig.tight_layout()
    
    # Convert to LLM format
    converter = FigureConverter()
    semantic_result = converter.convert(pair_plot, output_format="semantic")
    print_llm_analysis(semantic_result, 'pairplot')
    plt.close(pair_plot.fig)
    print("\n" + "="*50 + "\n")

def main():
    """Run all seaborn examples."""
    print("Seaborn Support Examples for plot2llm")
    print("=" * 50)
    
    try:
        # Run examples
        example_seaborn_scatter()
        example_seaborn_heatmap()
        example_seaborn_facetgrid()
        example_seaborn_distribution()
        example_seaborn_regression()
        example_seaborn_pairplot()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 