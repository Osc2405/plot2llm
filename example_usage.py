"""
Example usage of the plot2llm library.

This script demonstrates how to convert various types of matplotlib figures
into LLM-readable formats.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot2llm import FigureConverter


def create_simple_line_plot():
    """Create a simple line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, label='Sine Wave', linewidth=2, color='blue')
    ax.set_title('Simple Sine Wave Plot')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def create_scatter_plot():
    """Create a scatter plot with multiple series."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate random data
    np.random.seed(42)
    x1 = np.random.randn(50)
    y1 = np.random.randn(50)
    x2 = np.random.randn(50) + 2
    y2 = np.random.randn(50) + 2
    
    ax.scatter(x1, y1, label='Group A', alpha=0.6, color='red')
    ax.scatter(x2, y2, label='Group B', alpha=0.6, color='blue')
    
    ax.set_title('Scatter Plot: Two Groups')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    
    return fig


def create_bar_chart():
    """Create a bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    values = [23, 45, 56, 78, 32]
    
    bars = ax.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
    ax.set_title('Sample Bar Chart')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}', ha='center', va='bottom')
    
    return fig


def create_histogram():
    """Create a histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate random data
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    
    ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Histogram of Normal Distribution')
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    return fig


def create_multi_subplot():
    """Create a figure with multiple subplots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Subplot 1: Line plot
    x = np.linspace(0, 5, 100)
    ax1.plot(x, np.sin(x), label='sin(x)')
    ax1.plot(x, np.cos(x), label='cos(x)')
    ax1.set_title('Trigonometric Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot
    x_scatter = np.random.randn(30)
    y_scatter = np.random.randn(30)
    ax2.scatter(x_scatter, y_scatter, alpha=0.6)
    ax2.set_title('Random Scatter')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Bar plot
    categories = ['A', 'B', 'C']
    values = [10, 20, 15]
    ax3.bar(categories, values, color=['red', 'blue', 'green'])
    ax3.set_title('Simple Bar Chart')
    
    # Subplot 4: Histogram
    data_hist = np.random.normal(0, 1, 500)
    ax4.hist(data_hist, bins=20, alpha=0.7, color='orange')
    ax4.set_title('Data Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate the library."""
    print("Plot2LLM Library Demo")
    print("=" * 50)
    
    # Initialize the converter
    converter = FigureConverter()
    
    # Create different types of plots
    plots = {
        'Simple Line Plot': create_simple_line_plot(),
        'Scatter Plot': create_scatter_plot(),
        'Bar Chart': create_bar_chart(),
        'Histogram': create_histogram(),
        'Multi-subplot': create_multi_subplot()
    }
    
    # Convert each plot to different formats
    for plot_name, fig in plots.items():
        print(f"\n{plot_name}")
        print("-" * 30)
        
        # Convert to text format
        text_result = converter.convert(fig, output_format='text')
        print(f"Text format (first 200 chars): {text_result[:200]}...")
        
        # Convert to JSON format
        json_result = converter.convert(fig, output_format='json')
        print(f"JSON format (first 200 chars): {json_result[:200]}...")
        
        # Convert to semantic format
        semantic_result = converter.convert(fig, output_format='semantic')
        print(f"Semantic format (first 200 chars): {semantic_result[:200]}...")
        
        plt.close(fig)
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main() 