"""
Example demonstrating seaborn support in plot2llm.

This example shows how to use the library with various seaborn plot types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot2llm import FigureConverter, TextFormatter, JSONFormatter

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
    result = converter.convert(fig, "seaborn", output_format="text")
    
    print("Analysis Result:")
    print(result)
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
    result = converter.convert(fig, "seaborn", output_format="json")
    
    print("Analysis Result (JSON):")
    print(result)
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
    result = converter.convert(g, "seaborn", output_format="text")
    
    print("Analysis Result:")
    print(result)
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
    result = converter.convert(fig, "seaborn", output_format="text")
    
    print("Analysis Result:")
    print(result)
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
    result = converter.convert(fig, "seaborn", output_format="text")
    
    print("Analysis Result:")
    print(result)
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
    result = converter.convert(pair_plot, "seaborn", output_format="text")
    
    print("Analysis Result:")
    print(result)
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